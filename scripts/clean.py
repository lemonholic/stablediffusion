from shutil import rmtree


if __name__ == '__main__':
    try:
        rmtree('./outputs/txt2img-samples/samples')
    except FileNotFoundError:
        pass
