# scripts/analyze_image.py
import argparse
from puzzles_generator.image_processor import ImageAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze objects in an image')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--num-objects', type=int, default=20, 
                      help='Minimum number of objects to detect (default: 20)')
    args = parser.parse_args()
    
    analyzer = ImageAnalyzer()
    objects = analyzer.get_objects_from_image(args.image_path, num_objects=args.num_objects)
    
    print(f"\nObjects and related items found ({len(objects)} items):")
    for i, obj in enumerate(objects, 1):
        print(f"{i}. {obj}")

if __name__ == "__main__":
    main()