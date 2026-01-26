import re
import html
from typing import Dict, List, Tuple

import pandas as pd
import regex as regex
from ftfy import fix_text
from emoji import demojize, emojize
from unidecode import unidecode
import tldextract

try:
    import langid
except ImportError:
    langid = None


PRESERVE_CASE_FOR_CASED_MODELS = True

def detect_language(text: str) -> Tuple[str, float]:
    if langid is None:
        return("unknown", 0.0)
    
    code, score = langid.classify(text if text else "")
    return code, float(score)

def fix_unicode(text:str) -> str:
    return fix_text(text or "")

def strip_html(text: str) -> str:
    if not text:
        print('WARN: Empty text (strip_html)')
        return ""
    text = html.unescape(text)
    return re.sub(r"<[^>]+>", " ", text)

def normalize_whitespace(text: str) -> str:
    if not text:
        print('WARN: Empty text (normalize_whitespace)')
        return ""
    text = regex.sub(r"\s+", " ", text, flags=regex.MULTILINE).strip()
    return text

def remove_control_chars(text: str) -> str:
    '''
    Removes invisible controls
    '''

    if not text:
        print('WARN: Empty text (remove_control_chars)')
        return ""
    return regex.sub(r"[\p{C}&&[^\n\t]]+", "", text)

def extract_domain(url: str) -> str:
    ext = tldextract.extract(url)
    root = ".".join(part for part in [ext.domain, ext.suffix] if part)
    return root or (ext.fqdn or "URL")

def replace_urls(text: str, mode: str) -> str:
    '''
    Replaces URLs depending on the mode:
    token: all urls by 'URL'
    domain_token: all urls by the domain
    domain_visible: the visible domain between brackets
    keep: do nothing
    '''
    if not text:
        print('WARN: Empty text (replace_urls)')
        return ""

    url_re = re.compile(r"(https?|ftp)://[^\s]+", flags = re.IGNORECASE)

    def repl(m):
        url = m.group(0)
        domain = extract_domain(url)
        if mode == 'token':
            return "URL"
        elif mode == 'domain_token':
            return f'{domain.upper()}_TOKEN'
        elif mode == 'domain_visible':
            return f'[{domain}]'
        elif mode == 'keep':
            return url
        else:
            return 'URL'
        
    return url_re.sub(repl, text)
    
def replace_mentions(text: str, mode: str = 'token') -> str:
    '''
    Normalizes @mentions
    'token' -> @USER
    'keep' -> do nothing
    '''
    if not text:
        return ""
    if mode == 'keep':
        return text
    return re.sub(r"@\w+", "@USER", text)


def handle_emojis(text: str, mode: str, language: str = 'en'):
    '''
    'demojize': emoji to :smile:
    'keep' keeps the emoji as it is
    'strip' removes emojis
    '''

    if not text:
        return ""
    if mode == "demojize":
        return demojize(text, language=language)
    elif mode == "keep":
        return text
    elif mode == 'strip':
        s = text
        # ChatGPT kind of fixed this
        # 1) Preservar el dígito en secuencias keycap: (\d)\uFE0F?\u20E3  -> \1
        #    Ej.: '1️⃣' = '1' + U+FE0F (opc) + U+20E3 (COMBINING ENCLOSING KEYCAP)
        s = regex.sub(r"(\d)\uFE0F?\u20E3", r"\1", s)

        # 2) Eliminar emojis “pictográficos” sin tocar letras ni números normales.
        #    \p{Extended_Pictographic} cubre la mayoría de emojis modernos.
        s = regex.sub(r"\p{Extended_Pictographic}", "", s)

        # 3) También eliminamos otros símbolos de presentación emoji si quedaron.
        s = regex.sub(r"\p{Emoji_Presentation}", "", s)

        # 4) Quitar selectores de variación residuales, si los hubiera.
        s = s.replace("\uFE0F", "")

        # 5) (Opcional) Remover modificadores de tono de piel si aparecen sueltos.
        s = regex.sub(r"\p{Emoji_Modifier}", "", s)

        return s
    return text

def limit_punctuation_runs(text: str, max_run: int = 3) -> str:
    # limits punctuation (!!!!!)
    if not text:
        return ""
    return regex.sub(r"([!?.,;:])\1{%d,}" % (max_run), r"\1" * max_run, text)

def normalize_elongations(text: str, max_repeat: int = 3) -> str:
    '''
    Normalices letter elongation
    '''
    if not text:
        return ""
    return regex.sub(r"(.)\1{%d,}" % (max_repeat), r"\1" * max_repeat, text)


def make_lowercase(text: str, enabled: bool) -> str:
    if not text:
        return ""
    return text.lower() if enabled else text

def text_preprocessing(
        text: str,
        to_lowercase: bool,
        url_policy: str = 'token',
        mention_policy: str = 'token',
        emoji_policy: str = 'demojize',
        punctuation_runs: int = 3,
        elongation_runs: int = 3,    
) -> Dict[str, str]:
    s = fix_unicode(text)
    s = strip_html(s)
    s = remove_control_chars(s)
    s = normalize_whitespace(s)
    s = replace_urls(s, url_policy)
    s = replace_mentions(s, mention_policy)
    lang, conf = detect_language(s)
    try:
        s = handle_emojis(s, emoji_policy, lang)
    except Exception as e:
        # print(f'Error with handle_emojis: {e}')
        pass
    return {'text': s, 'lang': lang, "lang_conf": str(conf)}

def text_preprocessing_ml(text: str, to_lowercase: bool = True) -> Dict[str, str]:
    return text_preprocessing(text, to_lowercase, 'token', 'token', 'strip')

def text_preprocessing_llm(text: str, to_lowercase: bool = True) -> Dict[str, str]:
    return text_preprocessing(text, to_lowercase, 'domain_token', 'token', 'demojize')
