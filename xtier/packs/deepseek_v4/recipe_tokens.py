"""DeepSeek V4.1 prompts tokenized the way DeepSeek's deepseek-recipe does (XTIER_DS_RECIPE_TOKENS, default on; the
1017 pipeline's DS41_PLAIN_CONTENT_TOKENS of 2026-10-21).

SGLang renders a V4.1 prompt to one string and runs tokenizer.encode over it. An added-token spelling written in a user
message, a tool result or a tool definition ("<｜User｜>", "</think>" in a file a coding agent reads) then becomes a
control token, and a long conversation is tokenized whole on every turn (568K tokens: 1.3 s). deepseek-recipe treats
only the template's control tokens, image placeholders and the grounding tokens inside assistant content / reasoning
as special: here content spellings are swapped for private-use sentinels while rendering, the prompt is split at the
remaining (template) added tokens, and the plain regions, sentinels restored, are encoded as one batch with a copy of
the tokenizer that has no added vocabulary (0.18 s for the same text). Content without added-token spellings
tokenizes exactly as before.

Serving-side hooks, installed in the process that tokenizes (not in the schedulers): ``encoding_dsv41.encode_messages``
returns the prompt text as a ``RecipePrompt``, a str that carries its ids, and the serving tokenizer's ``encode``
returns those ids for it. Whatever edits the text on the way (image placeholders) passes a plain str and gets the
single-string encoding; so do prompts whose escaped spellings do not all come back from the renderer.
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)
ENCODE_MESSAGES = 'sglang.srt.entrypoints.openai.encoding_dsv41.encode_messages'
GET_TOKENIZER = 'sglang.srt.utils.hf_transformers.tokenizer.get_tokenizer'
GROUNDING_TOKENS = ('<｜ref｜>', '<｜/ref｜>', '<｜box｜>', '<｜/box｜>',
                    '<｜point｜>', '<｜/point｜>', '<｜polygon｜>', '<｜/polygon｜>')
SENTINEL_OPEN, SENTINEL_CLOSE = '', ''
SENTINEL_RE = re.compile('([0-9]+)')
ASSISTANT_TEXT_KEYS = ('content', 'reasoning_content', 'reasoning')
_parts = {}
_serving = [None]   # the tokenizer whose encode knows RecipePrompt (the serving process has one)


def enabled():
    return os.environ.get('XTIER_DS_RECIPE_TOKENS', '1') != '0'


class RecipePrompt(str):
    """A rendered V4.1 prompt that carries its deepseek-recipe token ids."""
    ids = ()


def tokenizer_parts(tokenizer):
    """(plain tokenizer without added vocabulary, added spelling -> id, split pattern, grounding spellings)."""
    parts = _parts.get(id(tokenizer))
    if parts is None:
        from tokenizers import Tokenizer

        backend = getattr(tokenizer, 'backend_tokenizer', None) or tokenizer._tokenizer
        spec = json.loads(backend.to_str())
        spec['added_tokens'] = []
        plain = Tokenizer.from_str(json.dumps(spec))
        plain.no_truncation()
        plain.no_padding()
        added = dict(tokenizer.get_added_vocab())
        # leftmost-longest, as the tokenizer's own added-token split
        pattern = re.compile('|'.join(re.escape(t) for t in sorted(added, key=len, reverse=True)))
        parts = (plain, added, pattern, frozenset(GROUNDING_TOKENS) & frozenset(added))
        _parts[id(tokenizer)] = parts
    return parts


def _escape(value, pattern, keep, store, state):
    if isinstance(value, str):
        if SENTINEL_OPEN in value or SENTINEL_CLOSE in value:
            state['collision'] = True

        def repl(match):
            spelling = match.group(0)
            if spelling in keep:
                return spelling
            store.append(spelling)
            return f'{SENTINEL_OPEN}{len(store) - 1}{SENTINEL_CLOSE}'

        return pattern.sub(repl, value)
    if isinstance(value, list):
        return [_escape(v, pattern, keep, store, state) for v in value]
    if isinstance(value, tuple):
        return tuple(_escape(v, pattern, keep, store, state) for v in value)
    if isinstance(value, dict):
        return {k: _escape(v, pattern, keep, store, state) for k, v in value.items()}
    return value


def _escape_messages(messages, pattern, grounding, store, state):
    return [{k: _escape(v, pattern, grounding if msg.get('role') == 'assistant' and k in ASSISTANT_TEXT_KEYS else (),
                        store, state) for k, v in msg.items()} for msg in messages]


def to_ids(prompt, tokenizer, store):
    """(ids, text) of a rendered prompt whose escaped spellings (store) are sentinels; None if they do not all come
    back. Split at the template's added tokens, plain regions encoded as one batch without added vocabulary."""
    plain, added, pattern, _ = tokenizer_parts(tokenizer)
    ids, texts, slots, pieces = [], [], [], []
    restored = [0]

    def restore(text):
        def repl(match):
            restored[0] += 1
            return store[int(match.group(1))]

        return SENTINEL_RE.sub(repl, text) if store else text

    pos = 0
    for match in pattern.finditer(prompt):
        if match.start() > pos:
            text = restore(prompt[pos:match.start()])
            slots.append(len(ids))
            ids.append(None)
            texts.append(text)
            pieces.append(text)
        ids.append(added[match.group(0)])
        pieces.append(match.group(0))
        pos = match.end()
    if pos < len(prompt):
        text = restore(prompt[pos:])
        slots.append(len(ids))
        ids.append(None)
        texts.append(text)
        pieces.append(text)
    if restored[0] != len(store):
        return None
    if texts:
        for slot, enc in zip(slots, plain.encode_batch(texts, add_special_tokens=False)):
            ids[slot] = enc.ids
    flat = []
    for item in ids:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat, ''.join(pieces)


def _encode_messages(original, messages, *args, **kwargs):
    """AROUND encoding_dsv41.encode_messages: the prompt text as a RecipePrompt with its ids."""
    tokenizer = _serving[0]
    if tokenizer is None or len(args) > 1:   # no serving tokenizer here, or context passed positionally
        return original(messages, *args, **kwargs)
    _, _, pattern, grounding = tokenizer_parts(tokenizer)
    store, state = [], {'collision': False}
    safe = _escape_messages(messages, pattern, grounding, store, state)
    safe_kwargs = dict(kwargs)
    if kwargs.get('context'):
        safe_kwargs['context'] = _escape_messages(kwargs['context'], pattern, grounding, store, state)
    if state['collision']:   # the content already holds a sentinel character
        return original(messages, *args, **kwargs)
    if not store:   # nothing to escape: render the caller's messages unchanged
        safe, safe_kwargs = messages, kwargs
    out = original(safe, *args, **safe_kwargs)
    multi = bool(kwargs.get('return_multi_modal_data'))
    prompt, media = out if multi else (out, None)
    done = to_ids(prompt, tokenizer, store)
    if done is None:
        logger.warning('xtier: dsv41 recipe tokenization: escaped spellings did not all come back; '
                       'using the single-string encoding')
        return original(messages, *args, **kwargs)
    result = RecipePrompt(done[1])
    result.ids = done[0]
    return (result, media) if multi else result


def wrap(tokenizer):
    """Make tokenizer.encode return a RecipePrompt's ids (anything else: the tokenizer's own encode)."""
    if tokenizer is None or getattr(tokenizer, '_xtier_recipe', False) or not hasattr(tokenizer, 'get_added_vocab'):
        return tokenizer
    if '<｜User｜>' not in tokenizer.get_added_vocab():   # not DeepSeek's chat tokens: the tokenizer stays as it is
        return tokenizer
    encode = tokenizer.encode

    def recipe_encode(text, *args, **kwargs):
        if type(text) is RecipePrompt and not args and not kwargs:
            return list(text.ids)
        return encode(text, *args, **kwargs)

    tokenizer.encode = recipe_encode
    tokenizer._xtier_recipe = True
    _serving[0] = tokenizer
    return tokenizer


def _get_tokenizer(result, *args, **kwargs):
    """AFTER get_tokenizer."""
    return wrap(result)


def install_serving(registry, hook_type):
    if not enabled():
        return False
    registry.register(ENCODE_MESSAGES, _encode_messages, hook_type.AROUND)
    registry.register(GET_TOKENIZER, _get_tokenizer, hook_type.AFTER)
    return True
