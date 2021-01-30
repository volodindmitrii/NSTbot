from PIL import Image
import numpy as np
from aiogram.types.base import Integer
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import copy
import warnings

warnings.filterwarnings('ignore')


def im_to_tensor(image):
    return loader(image).unsqueeze(0).to(device, torch.float)


def tensor_to_im(tensor):
    return unloader(tensor.cpu().clone().squeeze(0))


content_image_name = 'lisa.jpg'
first_style_image_name = 'graf.jpg'
second_style_image_name = 'gog.jpg'

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

imsize = 128
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(imsize),
    transforms.ToTensor()]
)
unloader = transforms.ToPILImage()

feature_image = im_to_tensor(Image.open(content_image_name))
style_image = im_to_tensor(Image.open(first_style_image_name))
style_images = im_to_tensor(Image.open(first_style_image_name)), im_to_tensor(Image.open(second_style_image_name))

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn = models.vgg19(pretrained=True).features.to(device).eval()


class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default,
                               style='simple'):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
        else:
            raise RuntimeError('Unrecognized style type: {}'.format(style))

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=500,
                       style_weight=100000, content_weight=1, style='simple'):
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img,
                                                                     content_img, style=style)
    optimizer = get_input_optimizer(input_img)

    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()

    features = input.reshape(batch_size * h, w * f_map_num)

    G = torch.mm(features, features.t())

    return G.div(batch_size * h * w * f_map_num)


import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage


API_TOKEN = '1466048254:AAG0_7Cy3Nh2XmZ5udcZPOVOfshLsUUPG8Q'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot, storage=MemoryStorage())


@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    poll_keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    poll_keyboard.add(types.KeyboardButton(text='/simple'))
    poll_keyboard.add(types.KeyboardButton(text='/near'))
    poll_keyboard.add(types.KeyboardButton(text='/both'))
    await message.answer("Hi!\nI'm VolodinBot!\nHere you can use style transfer machine.\n"
                         "For it you have a command /simple", reply_markup=poll_keyboard)


class WaitingPhotos(StatesGroup):
    waiting_for_content_photo = State()
    waiting_for_style_photo = State()
    waiting_for_first_content_photo = State()
    waiting_for_second_content_photo = State()
    images_id = 0


@dp.message_handler(commands=['simple'], state='*')
async def work_with_simple(message: types.Message):
    remove_keyboard = types.ReplyKeyboardRemove()
    await message.answer("Now send me a content photo", reply_markup=remove_keyboard)
    await WaitingPhotos.waiting_for_content_photo.set()

id_for_images = 0


@dp.message_handler(content_types=types.ContentType.PHOTO,
                    state=WaitingPhotos.waiting_for_content_photo)
async def get_content_photo(message: types.Message):
    global id_for_images
    id_for_images = message.message_id
    await bot.download_file_by_id(message.photo[-1].file_id, 'content_photo_{}.jpg'.format(id_for_images))
    await message.answer("Good! So, now a style photo")
    await WaitingPhotos.waiting_for_style_photo.set()


@dp.message_handler(content_types=types.ContentType.PHOTO,
                    state=WaitingPhotos.waiting_for_style_photo)
async def get_style_photo(message: types.Message):
    global id_for_images
    await bot.download_file_by_id(message.photo[-1].file_id, 'style_photo_{}.jpg'.format(id_for_images))
    await message.answer("Yea! Wait for a result")
    await create_simple_image(message)


async def create_simple_image(message: types.Message):
    global id_for_images
    feature_image = im_to_tensor(Image.open('content_photo_{}.jpg'.format(id_for_images)))
    style_image = im_to_tensor(Image.open('style_photo_{}.jpg'.format(id_for_images)))

    input_image = feature_image.clone()
    output_simple = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                       feature_image, style_image, input_image, num_steps=100,
                                       style_weight=100000, content_weight=1)
    tensor_to_im(output_simple).save('saved_{}.jpg'.format(id_for_images), 'JPEG')

    await send_created_image(message)


async def send_created_image(message: types.Message):
    global id_for_images
    await message.answer_photo(photo=types.InputFile('saved_{}.jpg'.format(id_for_images)))


@dp.message_handler()
async def echo(message: types.Message):
    await message.answer(message.text)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
