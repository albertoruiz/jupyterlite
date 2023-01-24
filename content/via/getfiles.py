async def getfile(url,filename=None):
    from js import fetch
    
    if filename is None:
        filename = url.split('/')[-1]
    
    res = await fetch(url)
    the_bytes = (await (await res.blob()).arrayBuffer()).to_bytes()

    with open(filename,'wb') as file:
        ok = file.write(the_bytes)
    
    return res.ok, ok


async def remote_image(url):
    from js import fetch
    from io import BytesIO
    from PIL import Image

    res = await fetch(url)
    the_bytes = (await (await res.blob()).arrayBuffer()).to_bytes()
    image = Image.open(BytesIO(the_bytes))
    return np.array(image)


path = "https://raw.githubusercontent.com/albertoruiz/umucv/master/"
