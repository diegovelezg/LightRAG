# Contenido para: lightrag/llm/google.py

import os
import numpy as np
from typing import List, Union, AsyncIterator
import pipmaster as pm
import asyncio
import json

# 1. Instalador dinámico de la librería
if not pm.is_installed("google-generativeai"):
    pm.install("google-generativeai")
if not pm.is_installed("python-dotenv"):
    pm.install("python-dotenv")

import google.generativeai as genai
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from lightrag.utils import logger

# Carga las variables de entorno
load_dotenv(dotenv_path=".env", override=False)

# Función para verificar si una excepción es reintentable (errores de API, no de validación)
def is_retryable_google_api_error(exception):
    """Return True if the exception is a transient Google API error."""
    from google.api_core import exceptions
    return isinstance(exception, (
        exceptions.ResourceExhausted, # Rate limit
        exceptions.ServiceUnavailable, # Server error
        exceptions.InternalServerError,
        exceptions.DeadlineExceeded
    ))

# 2. Decorador de reintentos
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(is_retryable_google_api_error)
)
async def google_embed(
    texts: List[str],
    model: str,
    api_key: str = None,
    **kwargs,
) -> np.ndarray:
    """
    Genera embeddings de forma ASÍNCRONA para una lista de textos usando la API de Google.
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        raise ValueError("GOOGLE_API_KEY is required for Google embeddings.")

    try:
        genai.configure(api_key=api_key)
        
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            logger.warning("Input texts list is empty or contains only empty strings.")
            embedding_dim = 768
            return np.empty((0, embedding_dim), dtype=np.float32)

        logger.debug(f"Requesting Google embeddings for {len(valid_texts)} texts with model {model}")
        
        # Ejecuta la llamada síncrona en un hilo separado para no bloquear el event loop
        result = await asyncio.to_thread(
            genai.embed_content,
            model=model,
            content=valid_texts,
            task_type="retrieval_document"
        )
        
        embeddings = np.array(result['embedding'], dtype=np.float32)
        logger.debug(f"Successfully received embeddings of shape {embeddings.shape}")
        
        return embeddings

    except Exception as e:
        logger.error(f"Google AI embedding failed: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(is_retryable_google_api_error)
)
async def google_complete_async(
    prompt: str,
    model: str,
    system_prompt: str = None,
    history_messages: list = None,
    stream: bool = False,
    api_key: str = None,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    """
    Genera una respuesta de texto usando un modelo de Google (Gemini).
    """
    if not api_key:
        api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("GOOGLE_API_KEY environment variable not set.")
        raise ValueError("GOOGLE_API_KEY is required for Google LLM.")

    genai.configure(api_key=api_key)

    # --- INICIO DE LA MODIFICACIÓN ---
    # Extrae los parámetros de configuración de los kwargs
    generation_config = {
        "temperature": kwargs.pop("temperature", None),
        "top_p": kwargs.pop("top_p", None),
        "top_k": kwargs.pop("top_k", None),
        "max_output_tokens": kwargs.pop("max_output_tokens", None),
        "stop_sequences": kwargs.pop("stop", None)
    }
    # Filtra los valores que no son None para no enviar parámetros vacíos
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    safety_settings = kwargs.pop("safety_settings", None)
    
    # Convierte safety_settings de string JSON a objeto si es necesario
    if isinstance(safety_settings, str):
        try:
            safety_settings = json.loads(safety_settings)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format for safety_settings: {safety_settings}")
            safety_settings = None


    # Inicializa el modelo con la nueva configuración
    gemini_model = genai.GenerativeModel(
        model, 
        system_instruction=system_prompt,
        generation_config=generation_config if generation_config else None,
        safety_settings=safety_settings if safety_settings else None
    )
    # --- FIN DE LA MODIFICACIÓN ---

    chat_history = []
    if history_messages:
        for msg in history_messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            chat_history.append({"role": role, "parts": [msg.get("content", "")]})

    logger.debug(f"Sending request to Google Gemini model {model}")

    try:
        if stream:
            response = await gemini_model.generate_content_async(
                contents=[*chat_history, {"role": "user", "parts": [prompt]}], 
                stream=True
            )

            async def stream_generator():
                async for chunk in response:
                    if hasattr(chunk, 'text'):
                        yield chunk.text
            
            return stream_generator()
        else:
            response = await gemini_model.generate_content_async(
                contents=[*chat_history, {"role": "user", "parts": [prompt]}]
            )
            return response.text
    except Exception as e:
        logger.error(f"Google AI completion failed: {e}")
        raise
