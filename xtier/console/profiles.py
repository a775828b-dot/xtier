"""Named launch presets, kept in the console state directory rather than the repository."""
import json
import time
import uuid

from .core import _write_json


class Profiles:
    """Store independent snapshots; reading a preset never starts or stops a model."""

    def __init__(self, console):
        self.console = console
        self.path = console.state_dir / 'xtier-presets.json'

    def list(self):
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        if not isinstance(data, list):
            raise ValueError('保存的配置文件格式无效')
        return data

    def validate(self, model, values):
        if model not in self.console.models:
            raise ValueError('配置对应的模型当前不可用')
        keys = set(self.console.defaults(model))
        if not isinstance(values, dict) or set(values) != keys:
            raise ValueError('配置项与当前模型不一致，请重新保存配置')
        return self.console.validate(model, values)

    def save(self, name, model, values):
        name = name.strip()
        if not name or len(name) > 80:
            raise ValueError('配置名称须为 1–80 个字符')
        values = self.validate(model, values)
        entries = self.list()
        entry = dict(id=uuid.uuid4().hex, name=name, model=model, options=values, created=time.time())
        entries.append(entry)
        _write_json(self.path, entries)
        return entry

    def load(self, key):
        for entry in self.list():
            if entry['id'] == key:
                return {**entry, 'options': self.validate(entry['model'], entry['options'])}
        raise ValueError('找不到保存的配置')
