// macos_systap.m — system-audio capture via a Core Audio process tap.
//
// Replaces ScreenCaptureKit for the far (system-audio) channel on
// macOS 14.6+. A process tap captures the global output mix without the
// Screen Recording permission — it uses the narrower "audio recording"
// consent (NSAudioCaptureUsageDescription) instead, which is the right
// trust posture for a confidential-meeting tool.
//
// Flow (mirrors Apple's "Capturing system audio with Core Audio taps"
// and insidegui/AudioCap):
//   1. CATapDescription — global stereo mixdown, unmuted (the operator
//      must still HEAR the call while we tap it).
//   2. AudioHardwareCreateProcessTap -> tap object id.
//   3. Aggregate device wrapping the tap (private, auto-start).
//   4. AudioDeviceCreateIOProcID + AudioDeviceStart; the IO proc gets
//      the tapped float audio and forwards mono frames to Rust.
//
// All CoreAudio symbols here are macOS 14.4+; the Rust side only calls
// this after a 14.6 version gate. The C ABI below is what Rust binds to.

#import <Foundation/Foundation.h>
#import <CoreAudio/CoreAudio.h>
#import <CoreAudio/AudioHardwareTapping.h>
#import <AudioToolbox/AudioToolbox.h>
#import <os/log.h>

// CATapDescription lives in CoreAudio on 14.4+. Declare the bits we use
// so the file compiles against SDKs whose headers vary; the runtime
// class is resolved by the ObjC runtime.
typedef NS_ENUM(NSInteger, KronoTapMute) { KronoTapUnmuted = 0 };

@interface CATapDescription : NSObject
- (instancetype)initStereoGlobalTapButExcludeProcesses:(NSArray *)processes;
@property (nonatomic, copy) NSUUID *UUID;
@property (nonatomic, assign) NSInteger muteBehavior;
@property (nonatomic, getter=isPrivate) BOOL privateTap;
@property (nonatomic, getter=isExclusive) BOOL exclusive;
@property (nonatomic, copy) NSString *name;
@end

// Rust-side callback: receives interleaved-then-averaged mono f32 frames.
typedef void (*krono_audio_cb)(void *ctx, const float *samples, int count);

typedef struct KronoSysTap {
    AudioObjectID tap;
    AudioDeviceID aggregate;
    AudioDeviceIOProcID io_proc;
    krono_audio_cb cb;
    void *ctx;
    int channels; // channels in the tap stream, for the IO proc to mix
} KronoSysTap;

static os_log_t krono_log(void) {
    static os_log_t l;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ l = os_log_create("com.simkinselgazar.krono", "systap"); });
    return l;
}

// The IO proc: tapped audio arrives in inInputData as float32. Average
// channels to mono and hand the frames to Rust.
static OSStatus krono_io_proc(AudioObjectID device, const AudioTimeStamp *now,
                              const AudioBufferList *inInputData,
                              const AudioTimeStamp *inputTime,
                              AudioBufferList *outOutputData,
                              const AudioTimeStamp *outputTime, void *clientData) {
    (void)device; (void)now; (void)inputTime; (void)outOutputData; (void)outputTime;
    KronoSysTap *t = (KronoSysTap *)clientData;
    if (!t || !t->cb || !inInputData || inInputData->mNumberBuffers == 0) return noErr;

    const AudioBuffer *buf = &inInputData->mBuffers[0];
    int chans = buf->mNumberChannels > 0 ? (int)buf->mNumberChannels : 1;
    const float *data = (const float *)buf->mData;
    if (!data) return noErr;
    int total = (int)(buf->mDataByteSize / sizeof(float));
    int frames = total / chans;
    if (frames <= 0) return noErr;

    // Interleaved float frames -> mono average.
    float *mono = (float *)malloc(sizeof(float) * frames);
    if (!mono) return noErr;
    for (int f = 0; f < frames; f++) {
        float sum = 0.0f;
        for (int c = 0; c < chans; c++) sum += data[f * chans + c];
        mono[f] = sum / (float)chans;
    }
    t->cb(t->ctx, mono, frames);
    free(mono);
    return noErr;
}

static NSString *krono_default_output_uid(void) {
    AudioObjectID dev = kAudioObjectUnknown;
    UInt32 size = sizeof(dev);
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal, kAudioObjectPropertyElementMain};
    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, 0, NULL, &size, &dev) != noErr)
        return nil;
    CFStringRef uid = NULL;
    size = sizeof(uid);
    addr.mSelector = kAudioDevicePropertyDeviceUID;
    if (AudioObjectGetPropertyData(dev, &addr, 0, NULL, &size, &uid) != noErr || !uid)
        return nil;
    return (__bridge_transfer NSString *)uid;
}

// Start the tap. On success returns an opaque handle and writes the
// tap's sample rate + channel count. Returns NULL on any failure (the
// caller falls back to mic-only). Every failure is logged.
void *krono_systap_start(krono_audio_cb cb, void *ctx, double *out_sample_rate,
                         int *out_channels) {
    @autoreleasepool {
        Class descClass = NSClassFromString(@"CATapDescription");
        if (!descClass) {
            os_log_error(krono_log(), "CATapDescription unavailable (needs macOS 14.4+)");
            return NULL;
        }
        CATapDescription *desc =
            [[descClass alloc] initStereoGlobalTapButExcludeProcesses:@[]];
        desc.name = @"Krono system audio";
        desc.UUID = [NSUUID UUID];
        desc.muteBehavior = KronoTapUnmuted; // operator still hears the call
        desc.privateTap = YES;

        AudioObjectID tap = kAudioObjectUnknown;
        OSStatus err = AudioHardwareCreateProcessTap(desc, &tap);
        if (err != noErr || tap == kAudioObjectUnknown) {
            os_log_error(krono_log(), "AudioHardwareCreateProcessTap failed: %d", (int)err);
            return NULL;
        }

        NSString *outUID = krono_default_output_uid();
        NSString *aggUID = [[NSUUID UUID] UUIDString];
        NSMutableDictionary *agg = [@{
            @"uid": aggUID,
            @"name": @"Krono Tap",
            @"private": @YES,
            @"stacked": @NO,
            @"tapautostart": @YES,
            @"taplist": @[ @{ @"uid": [desc.UUID UUIDString], @"drift": @YES } ],
        } mutableCopy];
        if (outUID) {
            agg[@"master"] = outUID;
            agg[@"subdevices"] = @[ @{ @"uid": outUID } ];
        }

        AudioDeviceID aggregate = kAudioObjectUnknown;
        err = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)agg, &aggregate);
        if (err != noErr || aggregate == kAudioObjectUnknown) {
            os_log_error(krono_log(), "AudioHardwareCreateAggregateDevice failed: %d", (int)err);
            AudioHardwareDestroyProcessTap(tap);
            return NULL;
        }

        // Read the tap stream format for sample rate + channel count.
        AudioStreamBasicDescription asbd = {0};
        UInt32 size = sizeof(asbd);
        AudioObjectPropertyAddress fmt = {kAudioTapPropertyFormat,
                                          kAudioObjectPropertyScopeGlobal,
                                          kAudioObjectPropertyElementMain};
        if (AudioObjectGetPropertyData(tap, &fmt, 0, NULL, &size, &asbd) != noErr ||
            asbd.mSampleRate <= 0) {
            os_log_error(krono_log(), "reading tap format failed");
            AudioHardwareDestroyAggregateDevice(aggregate);
            AudioHardwareDestroyProcessTap(tap);
            return NULL;
        }

        KronoSysTap *t = (KronoSysTap *)calloc(1, sizeof(KronoSysTap));
        t->tap = tap;
        t->aggregate = aggregate;
        t->cb = cb;
        t->ctx = ctx;
        t->channels = asbd.mChannelsPerFrame > 0 ? (int)asbd.mChannelsPerFrame : 2;

        err = AudioDeviceCreateIOProcID(aggregate, krono_io_proc, t, &t->io_proc);
        if (err != noErr || !t->io_proc) {
            os_log_error(krono_log(), "AudioDeviceCreateIOProcID failed: %d", (int)err);
            AudioHardwareDestroyAggregateDevice(aggregate);
            AudioHardwareDestroyProcessTap(tap);
            free(t);
            return NULL;
        }
        err = AudioDeviceStart(aggregate, t->io_proc);
        if (err != noErr) {
            os_log_error(krono_log(), "AudioDeviceStart failed: %d", (int)err);
            AudioDeviceDestroyIOProcID(aggregate, t->io_proc);
            AudioHardwareDestroyAggregateDevice(aggregate);
            AudioHardwareDestroyProcessTap(tap);
            free(t);
            return NULL;
        }

        if (out_sample_rate) *out_sample_rate = asbd.mSampleRate;
        if (out_channels) *out_channels = t->channels;
        os_log(krono_log(), "system-audio tap started: %.0f Hz, %d ch",
               asbd.mSampleRate, t->channels);
        return t;
    }
}

void krono_systap_stop(void *handle) {
    if (!handle) return;
    KronoSysTap *t = (KronoSysTap *)handle;
    if (t->io_proc) {
        AudioDeviceStop(t->aggregate, t->io_proc);
        AudioDeviceDestroyIOProcID(t->aggregate, t->io_proc);
    }
    if (t->aggregate != kAudioObjectUnknown) AudioHardwareDestroyAggregateDevice(t->aggregate);
    if (t->tap != kAudioObjectUnknown) AudioHardwareDestroyProcessTap(t->tap);
    free(t);
}
