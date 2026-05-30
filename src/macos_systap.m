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
#import <dlfcn.h>

// CATapDescription lives in CoreAudio on 14.4+. Declare the bits we use
// so the file compiles against SDKs whose headers vary; the runtime
// class is resolved by the ObjC runtime.
typedef NS_ENUM(NSInteger, KronoTapMute) { KronoTapUnmuted = 0 };

// Only the members we actually use. Setting properties that don't exist
// on the real class throws an unrecognized-selector exception, so we
// keep this to the documented essentials (init + UUID + muteBehavior)
// and guard the call site with @try/@catch.
@interface CATapDescription : NSObject
- (instancetype)initStereoGlobalTapButExcludeProcesses:(NSArray *)processes;
@property (nonatomic, copy) NSUUID *UUID;
@property (nonatomic, assign) NSInteger muteBehavior;
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
    unsigned long long frames_seen; // total non-empty frames forwarded to Rust
    unsigned long long io_calls;    // total IO-block invocations (incl. empty)
} KronoSysTap;

static os_log_t krono_log(void) {
    static os_log_t l;
    static dispatch_once_t once;
    dispatch_once(&once, ^{ l = os_log_create("com.simkinselgazar.krono", "systap"); });
    return l;
}

// Append a diagnostic line to a readable log file. os_log does not
// surface from the hardened, notarized app via `log show`, so a plain
// file at a known path is what actually lets us see why the tap behaves
// the way it does. Best-effort; never throws.
static void krono_diag(NSString *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    NSString *line = [[NSString alloc] initWithFormat:fmt arguments:args];
    va_end(args);
    os_log(krono_log(), "%{public}@", line);
    NSString *path = [NSHomeDirectory() stringByAppendingPathComponent:
        @"Library/Application Support/com.simkinselgazar.krono/systap.log"];
    FILE *f = fopen(path.UTF8String, "a");
    if (f) {
        fputs([[line stringByAppendingString:@"\n"] UTF8String], f);
        fclose(f);
    }
}

// Ensure the app holds the "audio recording" consent
// (kTCCServiceAudioCapture). THIS IS THE LOAD-BEARING STEP for system
// audio: a Core Audio process tap is created successfully and its IO
// proc is installed even with NO consent, but it then delivers only
// EMPTY buffers — exactly the "err=0 everywhere, 0 frames" symptom we
// saw. Unlike the microphone, there is no public API to request this
// permission; the only thing that works (per insidegui/AudioCap) is the
// private TCC framework. We dlopen it, preflight, and request — which
// shows the system prompt the first time. Returns 1 if authorized (or
// if we can't tell and should let the tap try), 0 if explicitly denied.
static int krono_ensure_audio_capture_permission(void) {
    void *tcc = dlopen("/System/Library/PrivateFrameworks/TCC.framework/TCC", RTLD_NOW);
    if (!tcc) {
        krono_diag(@"TCC.framework dlopen failed (%s) — proceeding blind", dlerror());
        return 1; // fail open: let the tap attempt proceed
    }
    typedef int (*PreflightFn)(CFStringRef, CFDictionaryRef);
    typedef void (*RequestFn)(CFStringRef, CFDictionaryRef, void (^)(BOOL));
    PreflightFn preflight = (PreflightFn)dlsym(tcc, "TCCAccessPreflight");
    RequestFn request = (RequestFn)dlsym(tcc, "TCCAccessRequest");
    CFStringRef service = CFSTR("kTCCServiceAudioCapture");

    if (preflight) {
        // Observed values: 0 = authorized, 1 = denied, 2 = undetermined.
        int status = preflight(service, NULL);
        krono_diag(@"TCC audio-capture preflight=%d (0=auth 1=denied 2=undetermined)", status);
        if (status == 0) return 1; // already granted — no prompt needed
    }
    if (!request) {
        krono_diag(@"TCCAccessRequest symbol missing — proceeding blind");
        return 1; // fail open
    }
    // Undetermined (first run) → this shows the prompt. Already-denied →
    // TCC returns the cached denial without re-prompting. We run on the
    // Rust capture thread (never the main thread), so blocking on the
    // semaphore while the system presents the prompt is safe.
    __block int granted = -1;
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    krono_diag(@"requesting audio-capture consent (system prompt if undetermined)...");
    request(service, NULL, ^(BOOL g) {
        granted = g ? 1 : 0;
        dispatch_semaphore_signal(sem);
    });
    long timedOut = dispatch_semaphore_wait(
        sem, dispatch_time(DISPATCH_TIME_NOW, (int64_t)120 * NSEC_PER_SEC));
    if (timedOut != 0) {
        krono_diag(@"audio-capture consent: no response in 120s — treating as denied");
        return 0;
    }
    krono_diag(@"audio-capture consent granted=%d", granted);
    return granted == 1 ? 1 : 0;
}

// Process one IO buffer of tapped audio: average channels to mono and
// hand the frames to Rust. No file I/O here (this runs on the realtime
// audio thread); frame accounting is logged later, at stop.
static void krono_process_input(KronoSysTap *t, const AudioBufferList *inInputData) {
    if (!t || !t->cb || !inInputData || inInputData->mNumberBuffers == 0) return;
    const AudioBuffer *buf = &inInputData->mBuffers[0];
    int chans = buf->mNumberChannels > 0 ? (int)buf->mNumberChannels : 1;
    const float *data = (const float *)buf->mData;
    if (!data) return;
    int total = (int)(buf->mDataByteSize / sizeof(float));
    int frames = total / chans;
    if (frames <= 0) return;

    float *mono = (float *)malloc(sizeof(float) * frames);
    if (!mono) return;
    for (int f = 0; f < frames; f++) {
        float sum = 0.0f;
        for (int c = 0; c < chans; c++) sum += data[f * chans + c];
        mono[f] = sum / (float)chans;
    }
    t->cb(t->ctx, mono, frames);
    free(mono);
    t->frames_seen += (unsigned long long)frames;
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
        krono_diag(@"--- systap start ---");
        // Consent first: without kTCCServiceAudioCapture the tap below
        // succeeds but delivers only empty buffers. Fail to mic-only
        // (NULL) + the system-audio warning if the user declines.
        if (!krono_ensure_audio_capture_permission()) {
            krono_diag(@"audio-capture permission denied — falling back to mic-only");
            return NULL;
        }
        Class descClass = NSClassFromString(@"CATapDescription");
        if (!descClass) {
            krono_diag(@"CATapDescription class not found (needs macOS 14.4+)");
            return NULL;
        }
        // Guard the ObjC setup: if a selector is wrong on this macOS the
        // runtime raises an exception, and an uncaught ObjC exception
        // aborts the whole app. Catch it and fall back to mic-only
        // (NULL) + the system-audio warning instead of crashing.
        CATapDescription *desc = nil;
        NSString *tapUUID = nil;
        @try {
            desc = [[descClass alloc] initStereoGlobalTapButExcludeProcesses:@[]];
            desc.UUID = [NSUUID UUID];
            desc.muteBehavior = KronoTapUnmuted; // operator still hears the call
            tapUUID = [desc.UUID UUIDString];
        } @catch (NSException *e) {
            krono_diag(@"CATapDescription setup raised %@: %@", e.name, e.reason);
            return NULL;
        }
        if (!desc || !tapUUID) {
            krono_diag(@"CATapDescription produced no usable tap UUID");
            return NULL;
        }
        krono_diag(@"CATapDescription ok, uuid=%@", tapUUID);

        AudioObjectID tap = kAudioObjectUnknown;
        OSStatus err = AudioHardwareCreateProcessTap(desc, &tap);
        krono_diag(@"AudioHardwareCreateProcessTap err=%d tap=%u", (int)err, (unsigned)tap);
        if (err != noErr || tap == kAudioObjectUnknown) {
            return NULL;
        }

        NSString *outUID = krono_default_output_uid();
        krono_diag(@"default output uid=%@", outUID ?: @"(nil)");
        NSString *aggUID = [[NSUUID UUID] UUIDString];
        NSMutableDictionary *agg = [@{
            @"uid": aggUID,
            @"name": @"Krono Tap",
            @"private": @YES,
            @"stacked": @NO,
            @"tapautostart": @YES,
            @"taplist": @[ @{ @"uid": tapUUID, @"drift": @YES } ],
        } mutableCopy];
        if (outUID) {
            agg[@"master"] = outUID;
            agg[@"subdevices"] = @[ @{ @"uid": outUID } ];
        }

        AudioDeviceID aggregate = kAudioObjectUnknown;
        err = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)agg, &aggregate);
        krono_diag(@"AudioHardwareCreateAggregateDevice err=%d agg=%u", (int)err, (unsigned)aggregate);
        if (err != noErr || aggregate == kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(tap);
            return NULL;
        }

        // Read the tap stream format for sample rate + channel count.
        AudioStreamBasicDescription asbd = {0};
        UInt32 size = sizeof(asbd);
        AudioObjectPropertyAddress fmt = {kAudioTapPropertyFormat,
                                          kAudioObjectPropertyScopeGlobal,
                                          kAudioObjectPropertyElementMain};
        OSStatus fmtErr = AudioObjectGetPropertyData(tap, &fmt, 0, NULL, &size, &asbd);
        krono_diag(@"tap format err=%d rate=%.0f ch=%u", (int)fmtErr, asbd.mSampleRate,
                   (unsigned)asbd.mChannelsPerFrame);
        if (fmtErr != noErr || asbd.mSampleRate <= 0) {
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

        // Use the block + dedicated serial queue variant. For a tap
        // aggregate device, this is the form that actually drives the IO
        // callback (the plain AudioDeviceCreateIOProcID set up cleanly but
        // never fired). Mirrors Apple's AudioCap sample.
        dispatch_queue_t ioQueue =
            dispatch_queue_create("com.simkinselgazar.krono.systap.io", DISPATCH_QUEUE_SERIAL);
        err = AudioDeviceCreateIOProcIDWithBlock(
            &t->io_proc, aggregate, ioQueue,
            ^(const AudioTimeStamp *inNow, const AudioBufferList *inInputData,
              const AudioTimeStamp *inInputTime, AudioBufferList *outOutputData,
              const AudioTimeStamp *inOutputTime) {
                (void)inNow; (void)inInputTime; (void)outOutputData; (void)inOutputTime;
                t->io_calls++; // count every fire, even empty, to tell
                               // "never fired" from "fired but silent"
                krono_process_input(t, inInputData);
            });
        krono_diag(@"AudioDeviceCreateIOProcIDWithBlock err=%d", (int)err);
        if (err != noErr || !t->io_proc) {
            AudioHardwareDestroyAggregateDevice(aggregate);
            AudioHardwareDestroyProcessTap(tap);
            free(t);
            return NULL;
        }
        err = AudioDeviceStart(aggregate, t->io_proc);
        krono_diag(@"AudioDeviceStart err=%d", (int)err);
        if (err != noErr) {
            AudioDeviceDestroyIOProcID(aggregate, t->io_proc);
            AudioHardwareDestroyAggregateDevice(aggregate);
            AudioHardwareDestroyProcessTap(tap);
            free(t);
            return NULL;
        }

        if (out_sample_rate) *out_sample_rate = asbd.mSampleRate;
        if (out_channels) *out_channels = t->channels;
        krono_diag(@"system-audio tap RUNNING: %.0f Hz, %d ch", asbd.mSampleRate, t->channels);
        return t;
    }
}

void krono_systap_stop(void *handle) {
    if (!handle) return;
    KronoSysTap *t = (KronoSysTap *)handle;
    krono_diag(@"systap stop: IO block fired %llu times, delivered %llu non-empty frames total",
               t->io_calls, t->frames_seen);
    if (t->io_proc) {
        AudioDeviceStop(t->aggregate, t->io_proc);
        AudioDeviceDestroyIOProcID(t->aggregate, t->io_proc);
    }
    if (t->aggregate != kAudioObjectUnknown) AudioHardwareDestroyAggregateDevice(t->aggregate);
    if (t->tap != kAudioObjectUnknown) AudioHardwareDestroyProcessTap(t->tap);
    free(t);
}
