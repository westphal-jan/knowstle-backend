from sentence_transformers import SentenceTransformer, util
import json
import logging
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
from lxml import html
import trafilatura

logger = logging.getLogger(__name__)


model = SentenceTransformer("paraphrase-distilroberta-base-v1")


@csrf_exempt
def index(request: HttpRequest):
    if (request.method != "POST"):
        return HttpResponse("Not available")
    body = json.loads(request.body)
    url = body["url"]
    logger.info(f"url: {url}")
    headers = {
        "User-Agent": "My User Agent 1.0",
    }
    response = requests.get(url, headers=headers)
    content = response.content
    mytree = html.fromstring(content)
    title = mytree.find(".//title").text
    logger.info(f"title: {title}")
    extracted = trafilatura.extract(mytree)
    embedding = model.encode(extracted, convert_to_tensor=True)
    logger.info(f"Embeddings: {len(embedding)}")

    response = {
        "url": url,
        "title": title,
        "embedding": embedding.tolist()
    }
    return JsonResponse(response)
