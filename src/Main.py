import os
import sys
sys.path.append(os.path.dirname(__file__) + os.sep + '..')

import pygame
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from src.model.Model import MobileNetV1

CHECKPOINT_PATH = "checkpoint.pt"  
NUM_CLASSES = 10
class_labels = ['AIRPLANE', 'APPLE', 'BANANA', 'CAR', 'CAT', 
                'DUCK', 'FISH', 'HAND', 'HOUSE', 'SOCCER BALL']

TEMP_IMAGE_PATH = "images/Temp.png"  

pygame.init()

screen_width, screen_height = 800, 400
screen = pygame.display.set_mode([screen_width, screen_height])
pygame.display.set_caption('Paint & Guess!')

draw_area = pygame.Rect(0, 0, 400, 400)

screen.fill('white', draw_area)

font_big = pygame.font.SysFont('comicsansms', 24)
font_small = pygame.font.SysFont('comicsansms', 16)

drawing = False
last_pos = None
guessing = False
guess_result = None

def guess():

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    
    try:
        image = Image.open(TEMP_IMAGE_PATH).convert('RGB')
    except Exception as e:
        print("Không mở được ảnh tạm:", e)
        return "ERROR"
    
    image = transform(image).unsqueeze(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MobileNetV1(ch_in=3, n_classes=NUM_CLASSES).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    except Exception as e:
        print("Không load được checkpoint:", e)
        return "NO CP"
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        probs = nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)
    return class_labels[predicted.item()]

while True:

    pygame.draw.rect(screen, (36, 54, 66), (400, 0, 400, 400))
    instructions = [
        "Left Click: Draw",
        "Right Click: Clear drawing",
        "Middle Click: Guess drawn image"
    ]

    for i, text in enumerate(instructions):
        instr_text = font_small.render(text, True, (118,181,197))
        screen.blit(instr_text, (420, 50 + i * 30))
    
    if guess_result:
        result_text = font_big.render("I THINK IT IS: " + guess_result, True, (226,241,231))
        result_rect = result_text.get_rect(center=(600, 300))
        screen.blit(result_text, result_rect)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        elif event.type == pygame.MOUSEMOTION:
            if drawing:
                mouse_position = pygame.mouse.get_pos()

                if last_pos is not None and mouse_position[0] < draw_area.width:
                    pygame.draw.line(screen, 'black', last_pos, mouse_position, 5)

                last_pos = mouse_position
                
        elif event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:
                if event.pos[0] < draw_area.width:
                    drawing = True
                    guessing = False

            elif event.button == 3:
                pygame.draw.rect(screen, 'white', draw_area)
                guessing = False

            elif event.button == 2:
                crop = screen.subsurface(draw_area).copy()
                pygame.image.save(crop, TEMP_IMAGE_PATH)
                print("Processing prediction ...")
                guess_result = guess()
                print("Predicted:", guess_result)
                guessing = True
                
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
            last_pos = None
            
    pygame.display.update()

pygame.quit()
