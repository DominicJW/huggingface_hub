import base64
import io
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, ContextManager, Dict, Generator, List, Optional, Union, overload

from requests import Response

from ._inference_types import ClassificationOutput, ConversationalOutput, ImageSegmentationOutput
from .constants import INFERENCE_ENDPOINT
from .utils import build_hf_headers, get_session, hf_raise_for_status, is_pillow_available
from .utils._typing import Literal


if TYPE_CHECKING:
    from PIL import Image

# Related resources:
#    https://huggingface.co/tasks
#    https://huggingface.co/docs/huggingface.js/inference/README
#    https://github.com/huggingface/huggingface.js/tree/main/packages/inference/src
#    https://github.com/huggingface/text-generation-inference/tree/main/clients/python
#    https://github.com/huggingface/text-generation-inference/blob/main/clients/python/text_generation/client.py
#    https://huggingface.slack.com/archives/C03E4DQ9LAJ/p1680169099087869

# TODO:
# - handle options? wait_for_model, use_gpu,... See list: https://github.com/huggingface/huggingface.js/blob/main/packages/inference/src/types.ts#L1
# - handle parameters? we can based implementation on inference.js
# - validate inputs/options/parameters? with Pydantic for instance? or only optionally?
# - add all tasks
# - handle async requests
# - if a user tries to call a task on a model that doesn't support it, I'll gracefully handle the error to print to the user the available task(s) for their model.
#       invalid task: client.summarization(EXAMPLE, model="codenamewei/speech-to-text")


RECOMMENDED_MODELS = {
    "audio-classification": "superb/hubert-large-superb-er",
    "automatic-speech-recognition": "facebook/wav2vec2-large-960h-lv60-self",
    "conversational": "microsoft/DialoGPT-large",
    "image-classification": "google/vit-base-patch16-224",
    "image-segmentation": "facebook/detr-resnet-50-panoptic",
    "image-to-image": "timbrooks/instruct-pix2pix",
    "summarization": "facebook/bart-large-cnn",
    "text-to-speech": "espnet/kan-bayashi_ljspeech_vits",
}

UrlT = str
PathT = Union[str, Path]
BinaryT = Union[bytes, BinaryIO]
ContentT = Union[BinaryT, PathT, UrlT]


class InferenceClient:
    def __init__(
        self, model: Optional[str] = None, token: Optional[str] = None, timeout: Optional[int] = None
    ) -> None:
        # If set, `model` can be either a repo_id on the Hub or an endpoint URL.
        self.model: Optional[str] = model
        self.headers = build_hf_headers(token=token)
        self.timeout = timeout

    def __repr__(self):
        return f"<InferenceClient(model='{self.model if self.model else ''}', timeout={self.timeout})>"

    def post(
        self,
        json: Optional[Union[str, Dict, List]] = None,
        data: Optional[ContentT] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
    ) -> Response:
        url = self._resolve_url(model, task)

        if data is not None and json is not None:
            warnings.warn("Ignoring `json` as `data` is passed as binary.")

        with _open_as_binary(data) as data_as_binary:
            response = get_session().post(
                url, json=json, data=data_as_binary, headers=self.headers, timeout=self.timeout
            )
        hf_raise_for_status(response)
        return response

    def audio_classification(
        self,
        audio: ContentT,
        model: Optional[str] = None,
    ) -> ClassificationOutput:
        # Recommended: superb/hubert-large-superb-er
        response = self.post(data=audio, model=model, task="audio-classification")
        return response.json()

    def automatic_speech_recognition(
        self,
        audio: ContentT,
        model: Optional[str] = None,
    ) -> str:
        # Recommended: facebook/wav2vec2-large-960h-lv60-self
        response = self.post(data=audio, model=model, task="automatic-speech-recognition")
        return response.json()["text"]

    def conversational(
        self,
        text: str,
        generated_responses: Optional[List[str]] = None,
        past_user_inputs: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> ConversationalOutput:
        # Recommended: microsoft/DialoGPT-large
        payload: Dict[str, Any] = {"inputs": {"text": text}}
        if generated_responses is not None:
            payload["inputs"]["generated_responses"] = generated_responses
        if past_user_inputs is not None:
            payload["inputs"]["past_user_inputs"] = past_user_inputs
        if parameters is not None:
            payload["parameters"] = parameters
        response = self.post(json=payload, model=model, task="conversational")
        return response.json()

    def image_classification(
        self,
        image: ContentT,
        model: Optional[str] = None,
    ) -> ClassificationOutput:
        # Recommended: google/vit-base-patch16-224
        response = self.post(data=image, model=model, task="image-classification")
        return response.json()

    def image_segmentation(
        self,
        image: ContentT,
        model: Optional[str] = None,
    ) -> List[ImageSegmentationOutput]:
        # Recommended: facebook/detr-resnet-50-panoptic

        # Segment
        response = self.post(data=image, model=model, task="image-segmentation")
        output = response.json()

        # Parse masks as PIL Image
        if not isinstance(output, list):
            raise ValueError(f"Server output must be a list. Got {type(output)}: {str(output)[:200]}...")
        for item in output:
            item["mask"] = _b64_to_image(item["mask"])
        return output

    def image_to_image(
        self,
        image: ContentT,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        strength: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[int] = None,
        guess_mode: Optional[bool] = None,
    ) -> "Image":
        # Recommended: timbrooks/instruct-pix2pix
        parameters = {
            "prompt": prompt,
            "strength": strength,
            "negative_prompt": negative_prompt,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "guess_mode": guess_mode,
        }
        if all(parameter is None for parameter in parameters.values()):
            # Either only an image to send => send as raw bytes
            self.post(data=image, model=model, task="image-to-image")
            data = image
            payload: Optional[Dict[str, Any]] = None
        else:
            # Or an image + some parameters => use base64 encoding
            data = None
            payload = {"inputs": _b64_encode(image)}
            for key, value in parameters.items():
                if value is not None:
                    payload[key] = value

        response = self.post(json=payload, data=data, model=model, task="image-to-image")
        return _response_to_image(response)

    def summarization(
        self,
        text: str,
        parameters: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> str:
        payload: Dict[str, Any] = {"inputs": text}
        if parameters is not None:
            payload["parameters"] = parameters
        response = self.post(json=payload, model=model, task="summarization")
        return response.json()[0]["summary_text"]

    def text_to_speech(self, text: str, model: Optional[str] = None) -> bytes:
        response = self.post(json={"inputs": text}, model=model, task="text-to-speech")
        return response.content

    def _resolve_url(self, model: Optional[str], task: Optional[str]) -> str:
        model = model or self.model

        # If model is already a URL, ignore `task` and return directly
        if model is not None and (model.startswith("http://") or model.startswith("https://")):
            return model

        # # If no model but task is set => fetch the recommended one for this task
        if model is None:
            if task is None:
                raise ValueError(
                    "You must specify at least a model (repo_id or URL) or a task, either when instantiating"
                    " `InferenceClient` or when making a request."
                )
            model = _get_recommended_model(task)

        # TODO: handle when task is feature-extraction / sentence-similarity
        #       i.e. the only case where a model has several useful tasks

        # Compute InferenceAPI url
        return f"{INFERENCE_ENDPOINT}/models/{model}"


def _get_recommended_model(task: str) -> str:
    if task in RECOMMENDED_MODELS:
        return RECOMMENDED_MODELS[task]
    raise NotImplementedError()


@overload
def _open_as_binary(content: ContentT) -> ContextManager[BinaryT]:
    ...  # means "if input is not None, output will not be None"


@overload
def _open_as_binary(content: Literal[None]) -> ContextManager[Literal[None]]:
    ...  # means "if input is None, output will be None"


@contextmanager  # type: ignore
def _open_as_binary(content: Optional[ContentT]) -> Generator[Optional[BinaryT], None, None]:
    """Open `content` as a binary file, either from a URL, a local path, or raw bytes.

    Do nothing if `content` is None,
    """
    # If content is a string => must be either a URL or a path
    if isinstance(content, str):
        if content.startswith("https://") or content.startswith("http://"):
            yield get_session().get(content).content  # TODO: retrieve as stream and pipe to post request ?
            return
        content = Path(content)
        if not content.exists():
            raise FileNotFoundError(
                f"File not found at {content}. If `data` is a string, it must either be a URL or a path to a local"
                " file. To pass raw content, please encode it as bytes first."
            )

    # If content is a Path => open it
    if isinstance(content, Path):
        with content.open("rb") as f:
            yield f
    else:
        # Otherwise: already a file-like object or None
        yield content


def _b64_encode(content: ContentT) -> str:
    with _open_as_binary(content) as data:
        data_as_bytes = data if isinstance(data, bytes) else data.read()
        return base64.b64encode(data_as_bytes).decode()


def _b64_to_image(encoded_image: str) -> "Image":
    """Parse a base64-encoded string into a PIL Image."""
    Image = _import_pil_image()
    return Image.open(io.BytesIO(base64.b64decode(encoded_image)))


def _response_to_image(response: Response) -> "Image":
    """Parse a Response object into a PIL Image.

    Expects the response body to be raw bytes. To deal with b64 encoded images, use `_b64_to_image` instead.
    """
    Image = _import_pil_image()
    return Image.open(io.BytesIO(response.content))


def _import_pil_image():
    """Make sure `PIL` is installed on the machine."""
    if not is_pillow_available():
        raise ImportError(
            "Please install Pillow to use deal with images (`pip install Pillow`). If you don't want the image to be"
            " post-processed, use `client.post(...)` and get the raw response from the server."
        )
    from PIL import Image

    return Image


if __name__ == "__main__":
    client = InferenceClient()

    # Text to speech to text
    audio = client.text_to_speech("Hello world")
    client.audio_classification(audio)
    client.automatic_speech_recognition(audio)

    # Image classification
    client.image_classification("cat.jpg")
    client.image_classification(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
    )

    # Image segmentation
    for item in client.image_segmentation("cat.jpg"):
        item["mask"].save(f"cat_{item['label']}_{item['score']}.jpg")

    # Image to image (instruct pix2pix)
    image = client.image_to_image("cat.jpg", prompt="turn the cat into a tiger")
    image.save("tiger.jpg")

    # Text summary
    client.summarization("The Eiffel tower...")

    # Chat
    output = client.conversational("Hi, who are you?")
    client.conversational(
        "Wow, that's scary!",
        generated_responses=output["conversation"]["generated_responses"],
        past_user_inputs=output["conversation"]["past_user_inputs"],
    )