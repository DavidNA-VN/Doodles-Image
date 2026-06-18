import sys
from pathlib import Path

import pygame
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.model.Model import MobileNetV1


CHECKPOINT_PATH = PROJECT_ROOT / "best_checkpoint.pt"
TEMP_IMAGE_PATH = PROJECT_ROOT / "images" / "Temp.png"

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
DRAW_AREA_SIZE = 400
BACKGROUND_COLOR = (36, 54, 66)
TEXT_COLOR = (118, 181, 197)
RESULT_COLOR = (226, 241, 231)


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def load_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_labels = checkpoint.get("classes")
    if not class_labels:
        raise ValueError("Checkpoint does not contain class metadata: 'classes'")

    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise ValueError("Checkpoint does not contain 'model_state_dict'")

    model = MobileNetV1(ch_in=3, n_classes=len(class_labels)).to(device)
    model.load_state_dict(model_state)
    model.eval()

    return model, class_labels, device


def predict(image_path, model, transform, class_labels, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(image)
        predicted_index = outputs.argmax(dim=1).item()

    return class_labels[predicted_index].upper()


def draw_sidebar(screen, font_small, font_big, guess_result):
    pygame.draw.rect(screen, BACKGROUND_COLOR, (DRAW_AREA_SIZE, 0, DRAW_AREA_SIZE, SCREEN_HEIGHT))

    instructions = [
        "Left Click: Draw",
        "Right Click: Clear drawing",
        "Middle Click: Guess drawn image",
    ]

    for i, text in enumerate(instructions):
        instr_text = font_small.render(text, True, TEXT_COLOR)
        screen.blit(instr_text, (420, 50 + i * 30))

    if guess_result:
        result_text = font_big.render("I THINK IT IS: " + guess_result, True, RESULT_COLOR)
        result_rect = result_text.get_rect(center=(600, 300))
        screen.blit(result_text, result_rect)


def run_pygame_loop(model, transform, class_labels, device):
    pygame.init()

    screen = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGHT])
    pygame.display.set_caption("Paint & Guess!")

    draw_area = pygame.Rect(0, 0, DRAW_AREA_SIZE, DRAW_AREA_SIZE)
    screen.fill("white", draw_area)

    font_big = pygame.font.SysFont("comicsansms", 24)
    font_small = pygame.font.SysFont("comicsansms", 16)

    drawing = False
    last_pos = None
    guess_result = None

    while True:
        draw_sidebar(screen, font_small, font_big, guess_result)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEMOTION and drawing:
                mouse_position = pygame.mouse.get_pos()

                if last_pos is not None and mouse_position[0] < draw_area.width:
                    pygame.draw.line(screen, "black", last_pos, mouse_position, 5)

                last_pos = mouse_position

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 and event.pos[0] < draw_area.width:
                    drawing = True

                elif event.button == 3:
                    pygame.draw.rect(screen, "white", draw_area)
                    guess_result = None

                elif event.button == 2:
                    TEMP_IMAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
                    crop = screen.subsurface(draw_area).copy()
                    pygame.image.save(crop, TEMP_IMAGE_PATH)

                    print("Processing prediction ...")
                    try:
                        guess_result = predict(TEMP_IMAGE_PATH, model, transform, class_labels, device)
                    except Exception as e:
                        print("Prediction failed:", e)
                        guess_result = "ERROR"
                    print("Predicted:", guess_result)

            elif event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_pos = None

        pygame.display.update()


def main():
    model, class_labels, device = load_model(CHECKPOINT_PATH)
    transform = build_transform()
    run_pygame_loop(model, transform, class_labels, device)


if __name__ == "__main__":
    main()
