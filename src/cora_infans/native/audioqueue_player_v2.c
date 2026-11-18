#include <AudioToolbox/AudioToolbox.h>
#include <CoreAudio/CoreAudio.h>
#include <CoreFoundation/CoreFoundation.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define PLAYER_VERSION "cora-audioqueue-player-v2"
#define COMPLETION_TIMEOUT_SECONDS 10.0

typedef struct {
    pthread_mutex_t lock;
    unsigned int callback_count;
} PlayerContext;

static void output_callback(
    void *user_data,
    AudioQueueRef queue,
    AudioQueueBufferRef buffer
) {
    (void)queue;
    (void)buffer;
    PlayerContext *context = (PlayerContext *)user_data;
    pthread_mutex_lock(&context->lock);
    context->callback_count += 1;
    pthread_mutex_unlock(&context->lock);
}

static double monotonic_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec + (double)value.tv_nsec / 1000000000.0;
}

static int require_status(const char *operation, OSStatus status) {
    if (status == noErr) {
        return 0;
    }
    fprintf(stderr, "%s failed with OSStatus %d\n", operation, (int)status);
    return 1;
}

static unsigned int read_callback_count(PlayerContext *context) {
    pthread_mutex_lock(&context->lock);
    unsigned int callback_count = context->callback_count;
    pthread_mutex_unlock(&context->lock);
    return callback_count;
}

static int current_device_matches(AudioQueueRef queue, CFStringRef expected_uid) {
    CFStringRef actual_uid = NULL;
    UInt32 size = sizeof(actual_uid);
    OSStatus status = AudioQueueGetProperty(
        queue,
        kAudioQueueProperty_CurrentDevice,
        &actual_uid,
        &size
    );
    if (require_status("AudioQueueGetProperty(CurrentDevice)", status)) {
        return 0;
    }
    if (actual_uid == NULL) {
        fprintf(stderr, "AudioQueue returned no current-device UID\n");
        return 0;
    }
    return CFStringCompare(actual_uid, expected_uid, 0) == kCFCompareEqualTo;
}

static int cfstring_to_utf8(CFStringRef value, char *output, size_t output_size) {
    if (value == NULL || output_size == 0) {
        return 0;
    }
    return CFStringGetCString(
        value,
        output,
        (CFIndex)output_size,
        kCFStringEncodingUTF8
    );
}

static void print_json_string(const char *value) {
    putchar('"');
    for (const unsigned char *cursor = (const unsigned char *)value;
         *cursor != '\0';
         cursor++) {
        switch (*cursor) {
            case '"':
                fputs("\\\"", stdout);
                break;
            case '\\':
                fputs("\\\\", stdout);
                break;
            case '\b':
                fputs("\\b", stdout);
                break;
            case '\f':
                fputs("\\f", stdout);
                break;
            case '\n':
                fputs("\\n", stdout);
                break;
            case '\r':
                fputs("\\r", stdout);
                break;
            case '\t':
                fputs("\\t", stdout);
                break;
            default:
                if (*cursor < 0x20) {
                    printf("\\u%04x", *cursor);
                } else {
                    putchar(*cursor);
                }
        }
    }
    putchar('"');
}

static UInt32 output_channel_count(AudioDeviceID device_id) {
    AudioObjectPropertyAddress address = {
        kAudioDevicePropertyStreamConfiguration,
        kAudioDevicePropertyScopeOutput,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    if (AudioObjectGetPropertyDataSize(device_id, &address, 0, NULL, &size) !=
            noErr ||
        size == 0) {
        return 0;
    }
    AudioBufferList *buffers = (AudioBufferList *)malloc(size);
    if (buffers == NULL) {
        return 0;
    }
    if (AudioObjectGetPropertyData(
            device_id,
            &address,
            0,
            NULL,
            &size,
            buffers
        ) != noErr) {
        free(buffers);
        return 0;
    }
    UInt32 channels = 0;
    for (UInt32 index = 0; index < buffers->mNumberBuffers; index++) {
        channels += buffers->mBuffers[index].mNumberChannels;
    }
    free(buffers);
    return channels;
}

static CFStringRef copy_device_string(
    AudioDeviceID device_id,
    AudioObjectPropertySelector selector
) {
    AudioObjectPropertyAddress address = {
        selector,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    CFStringRef value = NULL;
    UInt32 size = sizeof(value);
    if (AudioObjectGetPropertyData(
            device_id,
            &address,
            0,
            NULL,
            &size,
            &value
        ) != noErr) {
        return NULL;
    }
    return value;
}

static int list_output_devices(void) {
    AudioObjectPropertyAddress address = {
        kAudioHardwarePropertyDevices,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    UInt32 size = 0;
    if (require_status(
            "AudioObjectGetPropertyDataSize(Devices)",
            AudioObjectGetPropertyDataSize(
                kAudioObjectSystemObject,
                &address,
                0,
                NULL,
                &size
            )
        )) {
        return 1;
    }
    if (size == 0 || size % sizeof(AudioDeviceID) != 0) {
        fprintf(stderr, "CoreAudio returned an invalid device list\n");
        return 1;
    }
    AudioDeviceID *device_ids = (AudioDeviceID *)malloc(size);
    if (device_ids == NULL) {
        fprintf(stderr, "could not allocate CoreAudio device list\n");
        return 1;
    }
    if (require_status(
            "AudioObjectGetPropertyData(Devices)",
            AudioObjectGetPropertyData(
                kAudioObjectSystemObject,
                &address,
                0,
                NULL,
                &size,
                device_ids
            )
        )) {
        free(device_ids);
        return 1;
    }

    UInt32 device_count = size / sizeof(AudioDeviceID);
    UInt32 output_index = 0;
    int first = 1;
    fputs("{\"schema_version\":2,\"player_version\":\"" PLAYER_VERSION
          "\",\"enumeration_mode\":\"coreaudio_property_read_only\","
          "\"audio_queue_created\":false,\"audio_frames_sent\":0,"
          "\"outputs\":[",
          stdout);
    for (UInt32 index = 0; index < device_count; index++) {
        if (output_channel_count(device_ids[index]) == 0) {
            continue;
        }
        CFStringRef name = copy_device_string(
            device_ids[index],
            kAudioObjectPropertyName
        );
        CFStringRef uid = copy_device_string(
            device_ids[index],
            kAudioDevicePropertyDeviceUID
        );
        char name_text[1024];
        char uid_text[1024];
        int valid = cfstring_to_utf8(name, name_text, sizeof(name_text)) &&
                    cfstring_to_utf8(uid, uid_text, sizeof(uid_text));
        if (valid) {
            if (!first) {
                putchar(',');
            }
            printf("{\"index\":%u,\"name\":", output_index);
            print_json_string(name_text);
            fputs(",\"uid\":", stdout);
            print_json_string(uid_text);
            putchar('}');
            first = 0;
            output_index += 1;
        }
        if (name != NULL) {
            CFRelease(name);
        }
        if (uid != NULL) {
            CFRelease(uid);
        }
    }
    free(device_ids);
    fputs("]}\n", stdout);
    if (output_index == 0) {
        fprintf(stderr, "CoreAudio returned no output devices\n");
        return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--version") == 0) {
        printf("%s\n", PLAYER_VERSION);
        return 0;
    }
    if (argc == 2 && strcmp(argv[1], "--list-devices") == 0) {
        return list_output_devices();
    }
    if (argc != 3) {
        fprintf(stderr, "usage: %s WAV_PATH COREAUDIO_DEVICE_UID\n", argv[0]);
        return 64;
    }

    int result = 1;
    const char *wav_path = argv[1];
    const char *uid_text = argv[2];
    AudioFileID audio_file = NULL;
    AudioQueueRef queue = NULL;
    AudioQueueBufferRef buffer = NULL;
    CFURLRef file_url = NULL;
    CFStringRef device_uid = NULL;
    void *audio_bytes = NULL;
    PlayerContext context;
    memset(&context, 0, sizeof(context));
    pthread_mutex_init(&context.lock, NULL);

    file_url = CFURLCreateFromFileSystemRepresentation(
        kCFAllocatorDefault,
        (const UInt8 *)wav_path,
        (CFIndex)strlen(wav_path),
        false
    );
    if (file_url == NULL) {
        fprintf(stderr, "could not create a file URL for the WAV\n");
        goto cleanup;
    }
    if (require_status(
            "AudioFileOpenURL",
            AudioFileOpenURL(file_url, kAudioFileReadPermission, 0, &audio_file)
        )) {
        goto cleanup;
    }

    AudioStreamBasicDescription format;
    UInt32 format_size = sizeof(format);
    if (require_status(
            "AudioFileGetProperty(DataFormat)",
            AudioFileGetProperty(
                audio_file,
                kAudioFilePropertyDataFormat,
                &format_size,
                &format
            )
        )) {
        goto cleanup;
    }
    if (format.mFormatID != kAudioFormatLinearPCM ||
        format.mBytesPerFrame == 0 ||
        format.mSampleRate <= 0.0) {
        fprintf(stderr, "WAV must contain finite-rate linear PCM\n");
        goto cleanup;
    }

    UInt64 byte_count_64 = 0;
    UInt32 byte_count_size = sizeof(byte_count_64);
    if (require_status(
            "AudioFileGetProperty(AudioDataByteCount)",
            AudioFileGetProperty(
                audio_file,
                kAudioFilePropertyAudioDataByteCount,
                &byte_count_size,
                &byte_count_64
            )
        )) {
        goto cleanup;
    }
    if (byte_count_64 == 0 || byte_count_64 > UINT32_MAX) {
        fprintf(stderr, "WAV audio byte count is empty or too large\n");
        goto cleanup;
    }
    UInt32 byte_count = (UInt32)byte_count_64;
    audio_bytes = malloc(byte_count);
    if (audio_bytes == NULL) {
        fprintf(stderr, "could not allocate WAV audio bytes\n");
        goto cleanup;
    }
    UInt32 bytes_read = byte_count;
    if (require_status(
            "AudioFileReadBytes",
            AudioFileReadBytes(audio_file, false, 0, &bytes_read, audio_bytes)
        ) || bytes_read != byte_count) {
        fprintf(stderr, "WAV audio payload was truncated\n");
        goto cleanup;
    }

    device_uid = CFStringCreateWithCString(
        kCFAllocatorDefault,
        uid_text,
        kCFStringEncodingUTF8
    );
    if (device_uid == NULL) {
        fprintf(stderr, "could not create CoreAudio device UID\n");
        goto cleanup;
    }
    if (require_status(
            "AudioQueueNewOutput",
            AudioQueueNewOutput(
                &format,
                output_callback,
                &context,
                NULL,
                NULL,
                0,
                &queue
            )
        )) {
        goto cleanup;
    }
    CFStringRef uid_property = device_uid;
    if (require_status(
            "AudioQueueSetProperty(CurrentDevice)",
            AudioQueueSetProperty(
                queue,
                kAudioQueueProperty_CurrentDevice,
                &uid_property,
                sizeof(uid_property)
            )
        )) {
        goto cleanup;
    }
    if (!current_device_matches(queue, device_uid)) {
        fprintf(stderr, "AudioQueue current-device UID did not match request\n");
        goto cleanup;
    }
    if (require_status(
            "AudioQueueSetParameter(Volume)",
            AudioQueueSetParameter(queue, kAudioQueueParam_Volume, 1.0f)
        )) {
        goto cleanup;
    }
    if (require_status(
            "AudioQueueAllocateBuffer",
            AudioQueueAllocateBuffer(queue, byte_count, &buffer)
        )) {
        goto cleanup;
    }
    memcpy(buffer->mAudioData, audio_bytes, byte_count);
    buffer->mAudioDataByteSize = byte_count;
    if (require_status(
            "AudioQueueEnqueueBuffer",
            AudioQueueEnqueueBuffer(queue, buffer, 0, NULL)
        )) {
        goto cleanup;
    }

    UInt32 prepared_frames = 0;
    if (require_status(
            "AudioQueuePrime",
            AudioQueuePrime(queue, 0, &prepared_frames)
        )) {
        goto cleanup;
    }
    double started_at = monotonic_seconds();
    if (require_status("AudioQueueStart", AudioQueueStart(queue, NULL))) {
        goto cleanup;
    }

    unsigned int callback_count = 0;
    while (callback_count < 1) {
        callback_count = read_callback_count(&context);
        if (monotonic_seconds() - started_at > COMPLETION_TIMEOUT_SECONDS) {
            fprintf(stderr, "timed out waiting for returned-buffer callback\n");
            goto cleanup;
        }
        if (callback_count < 1) {
            usleep(10000);
        }
    }
    double callback_observed_at = monotonic_seconds();

    if (require_status("AudioQueueStop(non-immediate)", AudioQueueStop(queue, false))) {
        goto cleanup;
    }

    UInt32 running = 1;
    while (running != 0) {
        UInt32 running_size = sizeof(running);
        if (require_status(
                "AudioQueueGetProperty(IsRunning)",
                AudioQueueGetProperty(
                    queue,
                    kAudioQueueProperty_IsRunning,
                    &running,
                    &running_size
                )
            )) {
            goto cleanup;
        }
        if (monotonic_seconds() - started_at > COMPLETION_TIMEOUT_SECONDS) {
            fprintf(stderr, "timed out waiting for AudioQueue stop\n");
            goto cleanup;
        }
        if (running != 0) {
            usleep(10000);
        }
    }
    double stopped_at = monotonic_seconds();

    double duration_seconds =
        ((double)byte_count / (double)format.mBytesPerFrame) / format.mSampleRate;
    printf(
        "{\"schema_version\":2,\"player_version\":\"%s\","
        "\"requested_device_uid\":\"%s\",\"device_uid_verified\":true,"
        "\"audio_byte_count\":%u,\"sample_rate_hz\":%.0f,"
        "\"duration_seconds\":%.9f,\"prepared_frames\":%u,"
        "\"buffer_callback_count\":%u,"
        "\"callback_observed_before_stop\":true,"
        "\"stop_requested_after_callback\":true,"
        "\"queue_drain_confirmed\":true,"
        "\"callback_elapsed_seconds\":%.9f,"
        "\"playback_elapsed_seconds\":%.9f}\n",
        PLAYER_VERSION,
        uid_text,
        byte_count,
        format.mSampleRate,
        duration_seconds,
        prepared_frames,
        callback_count,
        callback_observed_at - started_at,
        stopped_at - started_at
    );
    result = 0;

cleanup:
    if (queue != NULL) {
        AudioQueueDispose(queue, result == 0 ? false : true);
    }
    if (audio_file != NULL) {
        AudioFileClose(audio_file);
    }
    if (device_uid != NULL) {
        CFRelease(device_uid);
    }
    if (file_url != NULL) {
        CFRelease(file_url);
    }
    free(audio_bytes);
    pthread_mutex_destroy(&context.lock);
    return result;
}
