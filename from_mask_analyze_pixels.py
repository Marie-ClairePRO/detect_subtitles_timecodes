import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os
import argparse

class SubtitleDetector:
    def __init__(self, 
                 edge_threshold = 80,
                 roi = [0, 0.5, 1., 0.5],
                 min_white_pixels=2000,
                 change_threshold=0.15,
                 spatial_consistency_threshold=0.7,
                 temporal_window=5,
                 min_subtitle_duration=0.4,
                 iou_consistency = 0.2,
                 print_process_bar = 100,
                 resize = True):
        """
        for text detection :
            edge_threshold : thrshold to select level of edge detection
            roi : region of interest, we will just take bottom half of the image by default to reduce zone of work

        for timecode detection : 
            min_white_pixels: Minimum white pixels to consider subtitle present
            change_threshold: Relative change threshold (0-1) to detect subtitle change
            spatial_consistency_threshold: IoU threshold for spatial consistency
            temporal_window: Number of frames to analyze for stability
            min_subtitle_duration: Minimum duration in seconds for valid subtitle
        """

        self.edge_threshold = edge_threshold
        self.roi = roi
        self.min_white_pixels_proportion = min_white_pixels
        self.change_threshold = change_threshold
        self.spatial_consistency_threshold = spatial_consistency_threshold
        self.temporal_window = temporal_window
        self.min_subtitle_duration = min_subtitle_duration
        self.iou_consistency = iou_consistency
        self.print_process_bar = print_process_bar
        self.resize = resize
        
    def init_cut_frame(self, frame_width, frame_height):
        # Convert normalized ROI to pixel coordinates
        div_size = 1
        if self.resize:
            div_size = int(np.ceil(frame_height*frame_width/1000000))
        x, y, w, h = self.roi
        self.x_px = int(x * frame_width / div_size)
        self.y_px = int(y * frame_height / div_size)
        self.w_px = int(w * frame_width / div_size)
        self.h_px = int(h * frame_height / div_size)
        self.min_white_pixels = int(self.h_px * self.w_px * self.min_white_pixels_proportion)
        return div_size

    def cut_frame(self, frame):
        return frame[self.y_px:self.y_px+self.h_px, self.x_px:self.x_px+self.w_px]

    def detect_subtitle_mask_structure(self, frame):
        """
        detect edges of image 
        return a black and white mask of the edges
        """
        half_frame = self.cut_frame(frame)
        lab = cv.cvtColor(half_frame, cv.COLOR_BGR2LAB)
        
        # Extract L channel from LAB - better for text detection
        l_channel = lab[:, :, 0]
        
        # Calculate gradients in both directions
        sobelx = cv.Sobel(l_channel, cv.CV_64F, 1, 0, ksize=3)
        sobely = cv.Sobel(l_channel, cv.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        maximum_gradient = gradient_mag.max()
        gradient_mag = np.uint8(gradient_mag / (maximum_gradient if maximum_gradient > 0 else 1) * 255)
        _, mask = cv.threshold(gradient_mag, self.edge_threshold, 255, cv.THRESH_BINARY)
        return mask

    def count_white_pixels(self, frame):
        return np.sum(frame > 200)
    
    def get_white_pixel_mask(self, frame):
        return (frame > 200).astype(np.uint8)
    
    def get_static_region_mask(self, current_mask, mask_history):
        """
        extract only the parts of the mask that didn't move by finding intersection
        of current frame with recent history
        return binary mask of static white regions
        """
        if len(mask_history) == 0:
            return current_mask
        
        # Start with current mask
        static_mask = current_mask.copy()
        
        # Intersect with all recent frames to keep only static regions
        for past_mask in mask_history:
            static_mask = np.logical_and(static_mask, past_mask)
        
        return static_mask.astype(np.uint8)
    
    def get_moving_region_mask(self, mask_history):
        """
        extract only the parts of the mask that didn't move by finding intersection
        of current frame with recent history
        return binary mask of static white regions
        """
        if len(mask_history) == 0:
            return None
        
        # Start with current mask
        moving_mask = mask_history[0].copy()
        
        # Intersect with all recent frames to keep only static regions
        for past_mask in mask_history:
            moving_mask = np.logical_or(moving_mask, past_mask)
        return moving_mask.astype(np.uint8)
    
    def compute_spatial_overlap(self, mask1, mask2):
        """
        compute Intersection over Union between two masks.
        high IoU means subtitles are in similar position.
        """
        if mask1 is None or mask2 is None:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 0.0
        
        #print(f"{intersection:.2f} / {union:.2f}  -------> {intersection/union:.2f}")
        
        return intersection / union
    
    def is_spatially_consistent(self, current_mask, mask_history):
        """
        check if current mask is spatially consistent with recent history.
        this filters out moving objects.
        """
        if len(mask_history) == 0:
            return True
        
        # compute consistency with recent masks
        overlaps = [self.compute_spatial_overlap(current_mask, past_mask) 
                   for past_mask in mask_history]
        
        avg_overlap = np.mean(overlaps)
        return avg_overlap >= self.spatial_consistency_threshold
    
    def detect_subtitle_changes(self, video_path, output_srt, verbose=True):
        """
        detect subtitle presence and changes from mask video.
        return dictionary with detection results and statistics
        """
        cap = cv.VideoCapture(video_path)
        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        div_size = self.init_cut_frame(frame_width=frame_width, frame_height=frame_height)
        
        print(f"Processing video: {video_path} ------ {frame_width}x{frame_height} @ {fps}fps, {total_frames} frames -- {int(total_frames/fps):.0f} seconds")
        if div_size > 1:
            print(f"Reduce size by {div_size} for faster processing -- {frame_width // div_size}x{frame_height // div_size}")
        print(f"ROI: {self.w_px}x{self.h_px} \n")
        
        white_pixel_counts = []
        static_pixel_counts = []
        timestamps = []
        mask_history = deque(maxlen=self.temporal_window)
        
        subtitle_segments = []
        current_segment = None
        
        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if div_size > 1:
                frame = cv.resize(frame, (frame_width // div_size, frame_height // div_size), interpolation=cv.INTER_AREA)
            current_mask = self.detect_subtitle_mask_structure(frame)
            
            # Count white pixels
            white_count = self.count_white_pixels(current_mask)
            white_pixel_counts.append(white_count)
            
            timestamp = frame_num / fps
            timestamps.append(timestamp)
            
            # Extract STATIC region only (intersection of recent frames)
            static_mask = self.get_static_region_mask(current_mask, list(mask_history))
            static_white_count = np.sum(static_mask)
            static_pixel_counts.append(static_white_count)
            
            has_subtitle = static_white_count >= self.min_white_pixels
            
            if has_subtitle:
                if current_segment is None:
                    # Start new segment
                    current_segment = {
                        'start_frame': max(0,frame_num-1),
                        'start_time': timestamp,
                        'start_pixels': static_white_count,
                        'pixel_counts': [static_white_count],
                        'frames': [frame_num]
                    }
                else:
                    # Continue segment
                    current_segment['pixel_counts'].append(static_white_count)
                    current_segment['frames'].append(frame_num)

                    # Extract all regions (union of recent frames) and check consistency within them
                    moving_mask = self.get_moving_region_mask(list(mask_history))
                    overlap = self.compute_spatial_overlap(moving_mask, current_mask)
                    is_consistent = overlap >= self.iou_consistency or overlap==0
                    
                    # Check for significant change in subtitle content (static and global)
                    avg_pixels = np.mean(current_segment['pixel_counts'][-self.temporal_window:])
                    relative_static_change = abs(static_white_count - avg_pixels) / (avg_pixels + 1)

                    avg_white_pixels = np.mean(white_pixel_counts[-len(current_segment['pixel_counts']):])
                    relative_white_change = abs(white_count - avg_white_pixels) / (avg_white_pixels + 1)

                    if False and is_consistent and (relative_static_change > self.change_threshold and relative_white_change > self.change_threshold):
                        print(timestamp, f"{overlap:02f}", f"{relative_static_change:02f}", f"{relative_white_change:02f}")
                    
                    if not is_consistent or (relative_static_change > self.change_threshold and relative_white_change > self.change_threshold):
                        # Subtitle changed -> end current segment (with duration check)
                        segment_duration = frame_num / fps - current_segment['start_time']
                        if segment_duration >= self.min_subtitle_duration:
                            current_segment['end_frame'] = frame_num
                            current_segment['end_time'] = frame_num / fps
                            current_segment['end_pixels'] = current_segment['pixel_counts'][-2]
                            subtitle_segments.append(current_segment)
                        
                        # Start new segment
                        current_segment = {
                            'start_frame': frame_num,
                            'start_time': timestamp,
                            'start_pixels': static_white_count,
                            'pixel_counts': [static_white_count],
                            'frames': [frame_num]
                        }
            else:
                # No valid subtitle
                if current_segment is not None:
                    # End current segment (with duration check)
                    segment_duration = frame_num / fps - current_segment['start_time']
                    
                    if segment_duration >= self.min_subtitle_duration:
                        current_segment['end_frame'] = frame_num
                        current_segment['end_time'] = frame_num / fps
                        current_segment['end_pixels'] = current_segment['pixel_counts'][-1]
                        subtitle_segments.append(current_segment)

                    #initialize new empty segment
                    current_segment = None

                mask_history.clear()
            
            # Add current segment to mask history for temporal analysis next step
            mask_history.append(current_mask)

            frame_num += 1
            if frame_num % self.print_process_bar == 0:
                print(f"Frames processed : {frame_num}/{total_frames}")
            #debug
            if frame_num == 3162:
                pass
        
        # Close last segment if exists (with duration check)
        if current_segment is not None:
            segment_duration = (frame_num - 1) / fps - current_segment['start_time']
            
            if segment_duration >= self.min_subtitle_duration:
                current_segment['end_frame'] = frame_num - 1
                current_segment['end_time'] = (frame_num - 1) / fps
                current_segment['end_pixels'] = current_segment['pixel_counts'][-1]
                subtitle_segments.append(current_segment)
        
        cap.release()
        
        results = {
            'white_pixel_counts': white_pixel_counts,
            'static_pixel_counts': static_pixel_counts,
            'frame_number': frame_num,
            'timestamps': timestamps,
            'subtitle_segments': subtitle_segments,
            'fps': fps
        }        
        # Plot analysis
        if verbose:
            self.plot_analysis(results)
        
        self.save_srt(results, output_srt)
        return results
    
    def print_summary(self, results):
        segments = results['subtitle_segments']
        print(f"\n{'='*60}")
        print(f"DETECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total subtitle segments detected: {len(segments)}")
        print(f"Total frames analyzed: {results['frame_number']}")
        print(f"\nSubtitle segments:")
        for i, seg in enumerate(segments, 1):
            print(f"  {i}. {self.seconds_to_timecode(seg['start_time'])} --> "
                  f"{self.seconds_to_timecode(seg['end_time'])} ")
        print(f"{'='*60}\n")
    
    def plot_analysis(self, results):        
        timestamps = np.array(results['timestamps'])
        white_counts = np.array(results['white_pixel_counts'])
        static_counts = np.array(results['static_pixel_counts'])

        t_length = 60
        plots = int(np.ceil(timestamps[-1]/t_length))
        _, axes = plt.subplots(plots, 1, figsize=(15, 8))

        if plots==1:
            axes = [axes]
        
        # White pixel count over time + static pixel counts over time
        for i in range(plots):
            start_t = i * t_length
            end_t = (i + 1) * t_length
            mask = (timestamps >= start_t) & (timestamps < end_t)

            axes[i].plot(timestamps[mask], white_counts[mask], linewidth=0.5, alpha=0.7)
            axes[i].plot(timestamps[mask], static_counts.clip(max=max(white_counts))[mask], linewidth=0.5, alpha=0.7, color='g')
            axes[i].axhline(y=self.min_white_pixels, color='r', linestyle='--', 
                        label=f'Threshold ({self.min_white_pixels} pixels)')
            axes[i].set_xlim(start_t, end_t)
            axes[i].set_xlabel('Time (seconds)')
            axes[i].set_ylabel('White Pixel Count')
            axes[i].set_title(f'White Pixel Count Over Time (Green = Detected Subtitles), Segment {i+1}: {start_t:.0f}s–{end_t:.0f}s')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Detected subtitle segments overlayed
        seg_starts = [seg['start_time'] for seg in results['subtitle_segments']]
        seg_ends = [seg['end_time'] for seg in results['subtitle_segments']]
        
        for h, (start, end) in enumerate(zip(seg_starts, seg_ends), 1):
            idx = int(start // t_length)
            axes[idx].barh(h, end - start, left=start, height=1000, 
                        color='green', alpha=0.7, edgecolor='black')
        
        plt.tight_layout()
        plt.savefig('subtitle_detection_analysis.png', dpi=150, bbox_inches='tight')
        print("Analysis plot saved as: subtitle_detection_analysis.png")
        plt.show()
    
    def seconds_to_timecode(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def save_srt(self, results, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(results['subtitle_segments'], 1):
                f.write(f"{i}\n")
                f.write(f"{self.seconds_to_timecode(seg['start_time'])} --> "
                       f"{self.seconds_to_timecode(seg['end_time'])}\n")
                f.write(f"{i} - {seg['end_time']-seg['start_time']:.2f}s\n\n")
        
        print(f"SRT file saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str, help="path to video with subs")
    parser.add_argument("--output_srt", type=str, default="detected_subtitles_timecodes.srt", help="srt file for output")
    parser.add_argument("--verbose", action='store_true', help="save detection analysis plot")
    args = parser.parse_args()

    # Initialize detector with parameters
    detector = SubtitleDetector(
        edge_threshold = 80,                  # only keep computed edges above a certain threshold 
        roi = [0., 0.5, 1., 0.5],             # region of interest : i take bottom half by default - [begin_width, begin_height, len_width, len_height]
        min_white_pixels=0.012,               # Minimum STATIC pixels to consider subtitle : 1.2% estimation of part in the image
        change_threshold=0.1,                 # **% change to detect new subtitle
        spatial_consistency_threshold=0.6,    # **% overlap for spatial consistency -> not used anymore (deprecated)
        temporal_window=5,                    # Analyze 5 frames for stability
        min_subtitle_duration=0.5,            # Minimum 0.4 seconds for valid subtitle
        iou_consistency=0.2,                  # consistency threshold for intersection over union sub mask
        print_process_bar=1200,               # print every *** frames processed for progression
        resize = True                         # if frame too big, reduce size until SD
    )
    
    # Process mask video
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
    
    results = detector.detect_subtitle_changes(
        video_path=args.video_path,
        output_srt=args.output_srt,
        verbose=args.verbose
    )
    
    print("\n✓ complete")
    print(f"Found {len(results['subtitle_segments'])} subtitle segments")