// macos_sckaudio.m — system-audio capture via ScreenCaptureKit.
//
// The far (system-audio) channel on macOS. We use ScreenCaptureKit's
// audio capture rather than a Core Audio process tap because the tap API
// (AudioHardwareCreateProcessTap + aggregate device) is too fragile on
// macOS 15/26 — for a *global* capture the aggregate's IO proc fires but
// the tap never surfaces as an input stream (mNumberBuffers == 0), and a
// tap-only aggregate refuses to start ('nope'). ScreenCaptureKit is the
// mature path shipping meeting apps use; it captures system audio
// reliably on current macOS.
//
// We do NOT record the screen. The stream is configured to the smallest
// allowable video size (2x2, 1 fps) and we never add a screen output
// handler nor read a single video frame — only the audio output is
// consumed. The Screen Recording ("Screen & System Audio Recording") TCC
// permission is required; startCapture triggers the system prompt the
// first time if it has not been granted.
//
// C ABI below mirrors the old tap shim so the Rust side (capture.rs) is
// a drop-in swap: krono_sckaudio_start / krono_sckaudio_stop.

#import <Foundation/Foundation.h>
#import <ScreenCaptureKit/ScreenCaptureKit.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreAudio/CoreAudio.h>
#import <AudioToolbox/AudioToolbox.h>

static const double KRONO_SCK_RATE = 48000.0;

// Rust-side callback: receives channel-averaged mono f32 frames.
typedef void (*krono_audio_cb)(void *ctx, const float *samples, int count);

// Append a diagnostic line to a readable log file. A hardened, notarized
// app's os_log output does not surface via `log show`, so a plain file at
// a known path is what lets us see why capture behaves the way it does.
static void krono_sck_diag(NSString *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    NSString *line = [[NSString alloc] initWithFormat:fmt arguments:args];
    va_end(args);
    NSString *path = [NSHomeDirectory() stringByAppendingPathComponent:
        @"Library/Application Support/com.simkinselgazar.krono/audio-capture.log"];
    FILE *f = fopen(path.UTF8String, "a");
    if (f) {
        fputs([[line stringByAppendingString:@"\n"] UTF8String], f);
        fclose(f);
    }
}

@interface KronoSCKCapture : NSObject <SCStreamOutput, SCStreamDelegate>
@property (nonatomic, assign) krono_audio_cb cb;
@property (nonatomic, assign) void *ctx;
@property (nonatomic, strong) SCStream *stream;
@property (nonatomic, strong) dispatch_queue_t audioQueue;
@property (nonatomic, assign) unsigned long long buffers_seen;
@property (nonatomic, assign) unsigned long long frames_seen;
@end

@implementation KronoSCKCapture

// Audio buffers arrive here on audioQueue. We mix every channel down to
// mono and forward the frames to Rust. No file I/O on this path; frame
// accounting is logged at stop.
- (void)stream:(SCStream *)stream
    didOutputSampleBuffer:(CMSampleBufferRef)sampleBuffer
                   ofType:(SCStreamOutputType)type {
    if (type != SCStreamOutputTypeAudio) return; // ignore video entirely
    if (!self.cb || !sampleBuffer) return;
    if (!CMSampleBufferDataIsReady(sampleBuffer)) return;

    CMFormatDescriptionRef fmt = CMSampleBufferGetFormatDescription(sampleBuffer);
    if (!fmt) return;
    const AudioStreamBasicDescription *asbd =
        CMAudioFormatDescriptionGetStreamBasicDescription(fmt);
    if (!asbd) return;
    int channels = asbd->mChannelsPerFrame > 0 ? (int)asbd->mChannelsPerFrame : 1;
    BOOL planar = (asbd->mFormatFlags & kAudioFormatFlagIsNonInterleaved) != 0;

    // Size the AudioBufferList, then fill it (retains a block buffer we
    // must release).
    size_t needed = 0;
    if (CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
            sampleBuffer, &needed, NULL, 0, NULL, NULL, 0, NULL) != noErr || needed == 0)
        return;
    AudioBufferList *abl = (AudioBufferList *)malloc(needed);
    if (!abl) return;
    CMBlockBufferRef block = NULL;
    OSStatus s = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(
        sampleBuffer, NULL, abl, needed, NULL, NULL, 0, &block);
    if (s != noErr || abl->mNumberBuffers == 0) {
        free(abl);
        if (block) CFRelease(block);
        return;
    }

    int nframes = 0;
    float *mono = NULL;
    if (planar) {
        // mNumberBuffers == channels; each buffer holds nframes mono floats.
        nframes = (int)(abl->mBuffers[0].mDataByteSize / sizeof(float));
        if (nframes > 0) {
            mono = (float *)malloc(sizeof(float) * nframes);
            if (mono) {
                for (int i = 0; i < nframes; i++) {
                    float sum = 0.0f; int nc = 0;
                    for (UInt32 b = 0; b < abl->mNumberBuffers; b++) {
                        const float *d = (const float *)abl->mBuffers[b].mData;
                        if (d) { sum += d[i]; nc++; }
                    }
                    mono[i] = nc > 0 ? sum / (float)nc : 0.0f;
                }
            }
        }
    } else {
        // One interleaved buffer of nframes*channels floats.
        const AudioBuffer *buf = &abl->mBuffers[0];
        const float *d = (const float *)buf->mData;
        int total = (int)(buf->mDataByteSize / sizeof(float));
        nframes = channels > 0 ? total / channels : 0;
        if (d && nframes > 0) {
            mono = (float *)malloc(sizeof(float) * nframes);
            if (mono) {
                for (int i = 0; i < nframes; i++) {
                    float sum = 0.0f;
                    for (int c = 0; c < channels; c++) sum += d[i * channels + c];
                    mono[i] = sum / (float)channels;
                }
            }
        }
    }

    if (mono && nframes > 0) {
        self.cb(self.ctx, mono, nframes);
        self.frames_seen += (unsigned long long)nframes;
        self.buffers_seen++;
    }
    if (mono) free(mono);
    free(abl);
    if (block) CFRelease(block);
}

- (void)stream:(SCStream *)stream didStopWithError:(NSError *)error {
    krono_sck_diag(@"stream stopped with error: %@", error.localizedDescription ?: @"(none)");
}

@end

// Start system-audio capture. On success returns an opaque retained
// handle and writes the configured sample rate + a channel count of 1
// (we deliver mono). Returns NULL on any failure (the caller falls back
// to mic-only + the system-audio warning). Blocks the calling thread
// (the Rust capture thread, never main) until capture has started.
void *krono_sckaudio_start(krono_audio_cb cb, void *ctx, double *out_sample_rate,
                           int *out_channels) {
    @autoreleasepool {
        krono_sck_diag(@"--- sckaudio start ---");
        if (@available(macOS 13.0, *)) {
            // ok
        } else {
            krono_sck_diag(@"ScreenCaptureKit audio needs macOS 13+");
            return NULL;
        }

        // 1. Shareable content (needs Screen Recording consent; triggers
        //    the prompt the first time). We only need a display to anchor
        //    the content filter — we never read its pixels.
        __block SCShareableContent *content = nil;
        __block NSError *contentErr = nil;
        dispatch_semaphore_t gotContent = dispatch_semaphore_create(0);
        [SCShareableContent getShareableContentWithCompletionHandler:^(
            SCShareableContent *c, NSError *e) {
            content = c;
            contentErr = e;
            dispatch_semaphore_signal(gotContent);
        }];
        dispatch_semaphore_wait(gotContent,
            dispatch_time(DISPATCH_TIME_NOW, (int64_t)15 * NSEC_PER_SEC));
        if (!content || content.displays.count == 0) {
            krono_sck_diag(@"no shareable content / no display (err=%@) — "
                           @"Screen Recording permission likely not granted",
                           contentErr.localizedDescription ?: @"(nil)");
            return NULL;
        }
        SCDisplay *display = content.displays.firstObject;
        krono_sck_diag(@"shareable content ok, %lu display(s)",
                       (unsigned long)content.displays.count);

        // 2. Content filter on the display, excluding nothing. The video
        //    is incidental; we capture it at the minimum size and never
        //    consume it.
        SCContentFilter *filter =
            [[SCContentFilter alloc] initWithDisplay:display excludingWindows:@[]];

        // 3. Audio-on, smallest-possible video. We never add a screen
        //    output, so no frame is ever delivered to us, but Screen
        //    CaptureKit still requires a video size — keep it at 2x2 / 1fps.
        SCStreamConfiguration *config = [[SCStreamConfiguration alloc] init];
        config.capturesAudio = YES;
        config.sampleRate = (NSInteger)KRONO_SCK_RATE;
        config.channelCount = 2;
        config.excludesCurrentProcessAudio = YES; // don't capture Krono itself
        config.width = 2;
        config.height = 2;
        config.minimumFrameInterval = CMTimeMake(1, 1); // 1 fps
        config.showsCursor = NO;
        config.queueDepth = 6;

        KronoSCKCapture *cap = [[KronoSCKCapture alloc] init];
        cap.cb = cb;
        cap.ctx = ctx;
        cap.audioQueue =
            dispatch_queue_create("com.simkinselgazar.krono.sckaudio", DISPATCH_QUEUE_SERIAL);

        SCStream *stream = [[SCStream alloc] initWithFilter:filter
                                              configuration:config
                                                   delegate:cap];
        cap.stream = stream;

        NSError *addErr = nil;
        BOOL added = [stream addStreamOutput:cap
                                        type:SCStreamOutputTypeAudio
                          sampleHandlerQueue:cap.audioQueue
                                       error:&addErr];
        if (!added) {
            krono_sck_diag(@"addStreamOutput(audio) failed: %@",
                           addErr.localizedDescription ?: @"(nil)");
            return NULL;
        }

        // 4. Start (async). Block until it confirms.
        __block NSError *startErr = nil;
        dispatch_semaphore_t started = dispatch_semaphore_create(0);
        [stream startCaptureWithCompletionHandler:^(NSError *e) {
            startErr = e;
            dispatch_semaphore_signal(started);
        }];
        long timedOut = dispatch_semaphore_wait(started,
            dispatch_time(DISPATCH_TIME_NOW, (int64_t)15 * NSEC_PER_SEC));
        if (timedOut != 0) {
            krono_sck_diag(@"startCapture timed out (no callback in 15s)");
            return NULL;
        }
        if (startErr) {
            krono_sck_diag(@"startCapture failed: %@", startErr.localizedDescription);
            return NULL;
        }

        if (out_sample_rate) *out_sample_rate = KRONO_SCK_RATE;
        if (out_channels) *out_channels = 1; // we deliver mono
        krono_sck_diag(@"system-audio capture RUNNING via ScreenCaptureKit: %.0f Hz",
                       KRONO_SCK_RATE);
        return (void *)CFBridgingRetain(cap);
    }
}

void krono_sckaudio_stop(void *handle) {
    if (!handle) return;
    @autoreleasepool {
        KronoSCKCapture *cap = (KronoSCKCapture *)CFBridgingRelease(handle);
        krono_sck_diag(@"sckaudio stop: %llu buffers, %llu frames total",
                       cap.buffers_seen, cap.frames_seen);
        if (cap.stream) {
            dispatch_semaphore_t stopped = dispatch_semaphore_create(0);
            [cap.stream stopCaptureWithCompletionHandler:^(NSError *e) {
                (void)e;
                dispatch_semaphore_signal(stopped);
            }];
            dispatch_semaphore_wait(stopped,
                dispatch_time(DISPATCH_TIME_NOW, (int64_t)5 * NSEC_PER_SEC));
        }
    }
}
