import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class ImageAnalyzer:
    def __init__(self):
        """Initialize the BLIP model and processor."""
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
    def get_objects_from_image(self, image_path, num_objects=20):
        """
        Analyze an image and return a list of potential objects and related items.
        
        Args:
            image_path (str): Path to the image file
            num_objects (int): Minimum number of objects to generate
            
        Returns:
            list: List of objects and related items found in or suggested by the image
        """
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        
        # List of prompts to generate diverse objects
        prompts = [
            "List all objects visible in this image:",
            "What items could be related to this scene?",
            "List the background objects in this image:",
            "What small details can be seen in this image?",
            "What weather-related items are associated with this scene?",
            "List any furniture or structures in this image:",
            "What natural elements are present in this scene?"
        ]
        
        all_objects = set()  # Use set to avoid duplicates
        
        for prompt in prompts:
            inputs = self.processor(image, prompt, return_tensors="pt")
            out = self.model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=3
            )
            
            for sequence in out:
                text = self.processor.decode(sequence, skip_special_tokens=True)
                # Split the text into individual items and clean them
                items = [item.strip().lower() for item in text.replace(',', '.').split('.')]
                # Add non-empty items to our set
                all_objects.update(item for item in items if item)
        
        # Convert set to sorted list
        object_list = sorted(list(all_objects))
        
        # If we don't have enough objects, generate more with additional prompts
        while len(object_list) < num_objects:
            additional_prompt = f"List more items you might find in a scene with {', '.join(object_list[:3])}:"
            inputs = self.processor(image, additional_prompt, return_tensors="pt")
            out = self.model.generate(
                **inputs,
                max_length=100,
                num_beams=5,
                do_sample=True,
                temperature=0.8,  # Slightly higher temperature for more variety
                num_return_sequences=2
            )
            
            for sequence in out:
                text = self.processor.decode(sequence, skip_special_tokens=True)
                items = [item.strip().lower() for item in text.replace(',', '.').split('.')]
                object_list.extend(item for item in items if item and item not in object_list)
            
            # Break if we're not getting any new items
            if len(object_list) >= num_objects or len(object_list) == len(all_objects):
                break
        
        return object_list[:num_objects] if len(object_list) > num_objects else object_list

# Example usage
if __name__ == "__main__":
    analyzer = ImageAnalyzer()
    
    # Example image path
    image_path = "path/to/your/image.jpg"
    
    # Get objects
    objects = analyzer.get_objects_from_image(image_path, num_objects=25)
    print("\nObjects and related items:")
    for i, obj in enumerate(objects, 1):
        print(f"{i}. {obj}")
