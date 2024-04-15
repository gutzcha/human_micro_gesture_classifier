# bash
mkdir SEGMENTED

mkdir SEGMENTED/PIS_ID_00
mkdir SEGMENTED/PIS_ID_00/Cam1
mkdir SEGMENTED/PIS_ID_00/Cam2
mkdir SEGMENTED/PIS_ID_00/Cam3
mkdir SEGMENTED/PIS_ID_00/Cam4
ffmpeg -i RAW/PIS_ID_000/Cam1/PIS_ID_00_2_Cam1_20200811_043527.036.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_00/Cam1/%4d.mp4
ffmpeg -i RAW/PIS_ID_000/Cam2/PIS_ID_00_2_Cam2_20200811_043527.036.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_00/Cam2/%4d.mp4
ffmpeg -i RAW/PIS_ID_000/Cam3/PIS_ID_00_2_Cam3_20200811_043527.036.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_00/Cam3/%4d.mp4
ffmpeg -i RAW/PIS_ID_000/Cam4/PIS_ID_00_2_Cam4_20200811_043527.036.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_00/Cam4/%4d.mp4

mkdir SEGMENTED/PIS_ID_02
mkdir SEGMENTED/PIS_ID_02/Cam1
mkdir SEGMENTED/PIS_ID_02/Cam2
mkdir SEGMENTED/PIS_ID_02/Cam3
mkdir SEGMENTED/PIS_ID_02/Cam4
ffmpeg -i RAW/PIS_ID_02/Cam1/PIS_ID_02_1_Cam1_20200811_053807.436.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_02/Cam1/%4d.mp4
ffmpeg -i RAW/PIS_ID_02/Cam2/PIS_ID_02_1_Cam2_20200811_053807.436.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_02/Cam2/%4d.mp4
ffmpeg -i RAW/PIS_ID_02/Cam3/PIS_ID_02_1_Cam3_20200811_053807.436.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_02/Cam3/%4d.mp4
ffmpeg -i RAW/PIS_ID_02/Cam4/PIS_ID_02_1_Cam4_20200811_053807.436.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_02/Cam4/%4d.mp4

mkdir SEGMENTED/PIS_ID_04
mkdir SEGMENTED/PIS_ID_04/Cam1
mkdir SEGMENTED/PIS_ID_04/Cam2
mkdir SEGMENTED/PIS_ID_04/Cam3
mkdir SEGMENTED/PIS_ID_04/Cam4
ffmpeg -i RAW/PIS_ID_04/Cam1/PIS_ID_04_1_Cam1_20200812_101103.513.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_04/Cam1/%4d.mp4
ffmpeg -i RAW/PIS_ID_04/Cam2/PIS_ID_04_1_Cam2_20200812_101103.513.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_04/Cam2/%4d.mp4
ffmpeg -i RAW/PIS_ID_04/Cam3/PIS_ID_04_1_Cam3_20200812_101103.513.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_04/Cam3/%4d.mp4
ffmpeg -i RAW/PIS_ID_04/Cam4/PIS_ID_04_1_Cam4_20200812_101103.513.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_04/Cam4/%4d.mp4

mkdir SEGMENTED/PIS_ID_05
mkdir SEGMENTED/PIS_ID_05/Cam1
mkdir SEGMENTED/PIS_ID_05/Cam2
mkdir SEGMENTED/PIS_ID_05/Cam3
mkdir SEGMENTED/PIS_ID_05/Cam4
ffmpeg -i RAW/PIS_ID_05/Cam1/PIS_ID_05_1_Cam1_20200812_120818.918.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_05/Cam1/%4d.mp4
ffmpeg -i RAW/PIS_ID_05/Cam2/PIS_ID_05_1_Cam2_20200812_120818.918.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_05/Cam2/%4d.mp4
ffmpeg -i RAW/PIS_ID_05/Cam3/PIS_ID_05_1_Cam3_20200812_120818.918.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_05/Cam3/%4d.mp4
ffmpeg -i RAW/PIS_ID_05/Cam4/PIS_ID_05_1_Cam4_20200812_120818.918.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_05/Cam4/%4d.mp4

mkdir SEGMENTED/PIS_ID_06
mkdir SEGMENTED/PIS_ID_06/Cam1
mkdir SEGMENTED/PIS_ID_06/Cam2
mkdir SEGMENTED/PIS_ID_06/Cam3
mkdir SEGMENTED/PIS_ID_06/Cam4
ffmpeg -i RAW/PIS_ID_06/Cam1/PIS_ID_06_1_Cam1_20200812_145659.779.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_06/Cam1/%4d.mp4
ffmpeg -i RAW/PIS_ID_06/Cam2/PIS_ID_06_1_Cam2_20200812_145659.779.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_06/Cam2/%4d.mp4
ffmpeg -i RAW/PIS_ID_06/Cam3/PIS_ID_06_1_Cam3_20200812_145659.779.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_06/Cam3/%4d.mp4
ffmpeg -i RAW/PIS_ID_06/Cam4/PIS_ID_06_1_Cam4_20200812_145659.779.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_06/Cam4/%4d.mp4

mkdir SEGMENTED/PIS_ID_07
mkdir SEGMENTED/PIS_ID_07/Cam1
mkdir SEGMENTED/PIS_ID_07/Cam2
mkdir SEGMENTED/PIS_ID_07/Cam3
mkdir SEGMENTED/PIS_ID_07/Cam4
ffmpeg -i RAW/PIS_ID_07/Cam1/PIS_ID_07_1_Cam1_20200812_170537.139.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_07/Cam1/%4d.mp4
ffmpeg -i RAW/PIS_ID_07/Cam2/PIS_ID_07_1_Cam2_20200812_170537.139.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_07/Cam2/%4d.mp4
ffmpeg -i RAW/PIS_ID_07/Cam3/PIS_ID_07_1_Cam3_20200812_170537.139.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_07/Cam3/%4d.mp4
ffmpeg -i RAW/PIS_ID_07/Cam4/PIS_ID_07_1_Cam4_20200812_170537.139.mp4 -c:v libx264 -crf 22 -map 0 -segment_time 2 -g 2 -sc_threshold 0 -force_key_frames "expr:gte(t,n_forced*2)" -reset_timestamps 1 -f segment SEGMENTED/PIS_ID_07/Cam4/%4d.mp4

