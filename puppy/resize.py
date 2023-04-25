from PIL import Image, ImageFilter
image = Image.open("puppy.jpg")

# resize
new_size = (980, 620)
new_image = Image.new("RGB", new_size)

# paste
resized_image = image.resize((720, 620))
new_image.paste(resized_image, ((new_size[0]-resized_image.size[0])//2, (new_size[1]-resized_image.size[1])//2))

# blur back
blurred_image = image.filter(ImageFilter.GaussianBlur(radius=50))
blurred_image = blurred_image.resize(new_size)
mask = Image.new('L', new_size, 255)
mask.paste(Image.new('L', resized_image.size, 0), ((new_size[0]-resized_image.size[0])//2, (new_size[1]-resized_image.size[1])//2))
new_image = Image.composite(blurred_image, new_image, mask)

# save
new_image.save("puppy.jpg")