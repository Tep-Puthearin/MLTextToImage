import tkinter as tk
from tkinter import font
import customtkinter as ctk

from PIL import ImageTk
from authtoken import auth_token

import torch
from torch import autocast
from diffusers import StableDiffusionPipeline

# Create the app
app = tk.Tk()
app.geometry("532x632")
app.title("Stable Bud")
ctk.set_appearance_mode("dark")

prompt = ctk.CTkEntry(height=40, width=512, text_color="black", fg_color="white", master=app)
prompt.configure(font=("Arial", 20))
prompt.place(x=10, y=10)

lmain = ctk.CTkLabel(master=app, height=512, width=512)
lmain.place(x=10, y=110)

modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"
pipe = StableDiffusionPipeline.from_pretrained(modelid, variant='fp16', torch_dtype=torch.float16)
pipe.to(device)


def generate():
    with autocast(device):
        image = pipe(prompt.get(), guidance_scale=8.5)
    image = image.images[0]
    img = ImageTk.PhotoImage(image)
    image.save('generatedimage.png')
    lmain.configure(image=img)
    lmain.image = img


trigger = ctk.CTkButton(height=40, width=120, text_color="white", fg_color="blue", master=app, command=generate)
trigger.configure(font=("Arial", 20), text="Generate")
trigger.place(x=206, y=60)

app.mainloop()