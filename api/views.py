from sentence_transformers import SentenceTransformer, util
import json
import logging
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
from lxml import html
import trafilatura
from urllib.parse import urlparse

logger = logging.getLogger(__name__)




@csrf_exempt
def index(request: HttpRequest):
    if (request.method != "POST"):
        return HttpResponse("Not available")
    model = SentenceTransformer("msmarco-MiniLM-L-6-v3")
    body = json.loads(request.body)
    url = body["url"]
    logger.info(f"url: {url}")
    headers = {
        "User-Agent": "My User Agent 1.0",
    }
    response = requests.get(url, headers=headers)
    response_content = response.content
    mytree = html.fromstring(response_content)
    title = mytree.find(".//title").text

    description = mytree.xpath("//meta[@name='description']/@content")
    logger.info(description)
    description = description[0] if description else ""
    image = mytree.xpath("//meta[@name='twitter:image']/@content")
    logger.info(image)
    image = image[0] if image else ""
    
    og_image = mytree.xpath("//meta[@property='og:image']/@content")
    og_image = og_image[0] if og_image else ""
    
    parsed_uri = urlparse(url)
    icon = "{uri.scheme}://{uri.netloc}/favicon.ico".format(uri=parsed_uri)

    # Trafilatura seems to modify the parsed tree and should therefore be called last or copied
    extracted = trafilatura.extract(mytree)
    embedding = model.encode(extracted, convert_to_tensor=True)

    response = {
        "url": url,
        "title": title,
        "description": description,
        "image": image,
        "og_image": og_image,
        "icon": icon,
    }
    logger.info(response)
    response["embedding"] = embedding.tolist()
    return JsonResponse(response)
