from torchvision import transforms

to_pil = transforms.ToPILImage()


def show_img(img):
    pic = to_pil(img)
    pic.show()
