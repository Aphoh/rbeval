import base64
import os
from pathlib import Path
import re
from typing import List


# Convenience methods from https://discuss.streamlit.io/t/image-in-markdown/13274/10
# Thanks random internet person!
def _markdown_images(markdown) -> List[str]:
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
    images = re.findall(
        r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))',
        markdown,
    )
    return images


def _img_to_bytes(img_path: str) -> str:
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def _img_to_html(img_path: str, img_alt: str) -> str:
    img_format = img_path.split(".")[-1]
    img_html = f'<img src="data:image/{img_format.lower()};base64,{_img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 70%;">'
    return img_html


def markdown_insert_images(markdown: str) -> str:
    images = _markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if os.path.exists(image_path):
            markdown = markdown.replace(
                image_markdown, _img_to_html(image_path, image_alt)
            )
    return markdown
