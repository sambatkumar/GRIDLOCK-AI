# detect.py
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import json

class HelmetDetector:
    def __init__(self, model_path='best.pt'):
        """
        Initialize helmet detector
        
        Args:
            model_path: Path to trained model or YOLOv8 model
        """
        # Load model
        self.model = YOLO(model_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        # Define class names and colors
        self.class_names = [
            'person', 'helmet', 'head-no-helmet', 
            'motorcycle', 'bicycle', 'safety_vest'
        ]
        
        # Colors for visualization (BGR format)
        self.colors = {
            'helmet': (0, 255, 0),          # Green - Safe
            'head-no-helmet': (0, 0, 255),  # Red - Violation
            'person': (255, 0, 0),          # Blue
            'motorcycle': (0, 255, 255),    # Yellow
            'bicycle': (255, 255, 0),       # Cyan
            'safety_vest': (255, 0, 255)    # Purple
        }
        
        # Violation tracking
        self.violations = []
        self.frame_count = 0
        
    def detect_frame(self, frame, conf_threshold=0.5, iou_threshold=0.5):
        """
        Detect helmets in a single frame
        
        Args:
            frame: Input image frame (numpy array)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Annotated frame and detection results
        """
        # Run inference
        results = self.model(
            frame, 
            conf=conf_threshold, 
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )
        
        # Process results
        detections = []
        annotated_frame = frame.copy()
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.cpu().numpy()
            
            for box in boxes:
                # Extract detection info
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = box.conf[0]
                class_id = int(box.cls[0])
                
                # Skip person class if we only care about helmets
                if class_id == 0:  # person
                    continue
                
                # Get class name and color
                class_name = self.class_names[class_id]
                color = self.colors.get(class_name, (255, 255, 255))
                
                # Store detection
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': class_name,
                    'area': (x2 - x1) * (y2 - y1)
                })
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Create label
                label = f"{class_name}: {confidence:.2f}"
                
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
        
        # Add statistics overlay
        annotated_frame = self.add_statistics_overlay(annotated_frame, detections)
        
        return annotated_frame, detections
    
    def add_statistics_overlay(self, frame, detections):
        """Add statistics overlay to frame"""
        height, width = frame.shape[:2]
        
        # Count detections by class
        counts = {name: 0 for name in self.class_names}
        for det in detections:
            counts[det['class_name']] += 1
        
        # Create statistics text
        stats = [
            f"Frame: {self.frame_count}",
            f"Helmets: {counts['helmet']}",
            f"No Helmets: {counts['head-no-helmet']} (VIOLATION)",
            f"Vehicles: {counts['motorcycle'] + counts['bicycle']}",
            f"Safety Vests: {counts['safety_vest']}"
        ]
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Draw statistics text
        for i, text in enumerate(stats):
            color = (0, 255, 0) if "VIOLATION" not in text else (0, 0, 255)
            y_pos = 40 + i * 25
            
            cv2.putText(
                frame,
                text,
                (20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (width - 300, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw legend
        legend_y = height - 100
        for i, (class_name, color) in enumerate(list(self.colors.items())[:4]):
            cv2.rectangle(frame, (20, legend_y + i*25), (50, legend_y + i*25 + 20), color, -1)
            cv2.putText(frame, class_name, (60, legend_y + i*25 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_image(self, image_path, output_dir='outputs'):
        """Process single image"""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        # Detect helmets
        annotated_frame, detections = self.detect_frame(frame)
        
        # Save output
        output_path = Path(output_dir) / f"detected_{Path(image_path).name}"
        cv2.imwrite(str(output_path), annotated_frame)
        
        # Save detection results
        results_path = Path(output_dir) / f"results_{Path(image_path).stem}.json"
        with open(results_path, 'w') as f:
            json.dump({
                'image_path': image_path,
                'detections': detections,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"Results saved to: {output_path}")
        print(f"Detection data saved to: {results_path}")
        
        # Display image
        cv2.imshow('Helmet Detection', annotated_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return detections
    
    def process_video(self, video_path, output_dir='outputs', show_live=False):
        """Process video file"""
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {video_path}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total frames: {total_frames}")
        
        # Prepare output video
        output_path = Path(output_dir) / f"detected_{Path(video_path).name}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Process video frame by frame
        frame_count = 0
        all_detections = []
        
        print("\nProcessing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, detections = self.detect_frame(frame)
            
            # Save frame to output video
            out.write(annotated_frame)
            
            # Track violations
            self.frame_count = frame_count
            helmet_detections = [d for d in detections if d['class_name'] == 'helmet']
            no_helmet_detections = [d for d in detections if d['class_name'] == 'head-no-helmet']
            
            if no_helmet_detections:
                violation = {
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'no_helmet_count': len(no_helmet_detections),
                    'helmet_count': len(helmet_detections)
                }
                self.violations.append(violation)
                all_detections.extend(detections)
            
            # Show live preview
            if show_live:
                cv2.imshow('Helmet Detection - Live', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"({frame_count/total_frames*100:.1f}%)")
        
        # Release resources
        cap.release()
        out.release()
        if show_live:
            cv2.destroyAllWindows()
        
        # Save video results
        self.save_violation_report(video_path, output_dir, all_detections)
        
        print(f"\nVideo processing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total violations detected: {len(self.violations)}")
        
        return all_detections
    
    def save_violation_report(self, video_path, output_dir, detections):
        """Save detailed violation report"""
        report_path = Path(output_dir) / f"report_{Path(video_path).stem}.json"
        
        report = {
            'video_path': str(video_path),
            'total_frames': self.frame_count,
            'total_violations': len(self.violations),
            'violations': self.violations,
            'detection_summary': {
                'total_detections': len(detections),
                'by_class': {}
            },
            'processing_time': datetime.now().isoformat()
        }
        
        # Count by class
        for class_name in self.class_names:
            count = len([d for d in detections if d['class_name'] == class_name])
            report['detection_summary']['by_class'][class_name] = count
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed report saved to: {report_path}")
    
    def real_time_detection(self, camera_id=0, conf_threshold=0.5):
        """Real-time helmet detection from webcam"""
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        print("\nReal-time Helmet Detection Started!")
        print("Press 'q' to quit")
        print("Press 's' to save current frame")
        print("Press 'v' to toggle violation alarm")
        
        alarm_on = True
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            annotated_frame, detections = self.detect_frame(frame, conf_threshold)
            self.frame_count = frame_count
            
            # Check for violations and trigger alarm
            no_helmet_detections = [d for d in detections if d['class_name'] == 'head-no-helmet']
            
            if no_helmet_detections and alarm_on:
                # Flash red border for violation
                cv2.rectangle(annotated_frame, (0, 0), 
                             (frame.shape[1], frame.shape[0]), 
                             (0, 0, 255), 10)
                
                # Add warning text
                cv2.putText(annotated_frame, "SAFETY VIOLATION DETECTED!", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (0, 0, 255), 3)
            
            # Show FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Real-time Helmet Detection', annotated_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{timestamp}.jpg", annotated_frame)
                print(f"Frame saved: capture_{timestamp}.jpg")
            elif key == ord('v'):
                alarm_on = not alarm_on
                status = "ON" if alarm_on else "OFF"
                print(f"Violation alarm turned {status}")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        print("\nReal-time detection stopped.")

def main():
    parser = argparse.ArgumentParser(description='Helmet Detection using YOLOv8')
    parser.add_argument('--mode', type=str, default='image',
                       choices=['image', 'video', 'realtime'],
                       help='Detection mode')
    parser.add_argument('--source', type=str, default='test.jpg',
                       help='Path to image/video or camera ID')
    parser.add_argument('--model', type=str, default='best.pt',
                       help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize detector
    print("Initializing Helmet Detector...")
    detector = HelmetDetector(args.model)
    print(f"Using device: {detector.device}")
    
    # Run detection based on mode
    if args.mode == 'image':
        print(f"Processing image: {args.source}")
        detector.process_image(args.source, args.output)
    
    elif args.mode == 'video':
        print(f"Processing video: {args.source}")
        detector.process_video(args.source, args.output, show_live=True)
    
    elif args.mode == 'realtime':
        print("Starting real-time detection from camera...")
        camera_id = 0 if args.source.isdigit() else args.source
        detector.real_time_detection(int(camera_id), args.conf)

if __name__ == "__main__":
    main()