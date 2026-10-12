from transformers import CLIPModel, CLIPProcessor

def main():
    clip_path = "./models/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.save_pretrained(clip_path)
    processor.save_pretrained(clip_path)

if __name__ == "__main__":
    main()
