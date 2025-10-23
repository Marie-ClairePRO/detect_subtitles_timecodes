# detect_subtitles_timecodes
Detect the timecodes of the subtitles in a video to process OCR on only one frame.

Returns a lot of false positive, but false negative are rare.

It is useful for processing OCR on videos and reduce time processing and GPU power by anaysing most frames of one video.

You can then launch for each chunk of timecode the OCR of the middle frame and insert its result in the srt file.

Opensource AIs like qwen-vl are available and very powerful on these tasks.
