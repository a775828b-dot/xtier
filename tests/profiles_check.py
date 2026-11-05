"""Preset and accepted-apply snapshot regressions without model processes."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xtier.console.core import Console
from xtier.console.profiles import Profiles
from xtier.console.__main__ import Jobs


class ProfilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        models = {'demo': {'options': [{'key': 'threads', 'label': 'Threads', 'default': '4',
                                        'choices': [('4', '4'), ('8', '8')]}],
                           'build': lambda values, port: ({}, ['server']), 'port': 1234}}
        self.console = Console(models, self.tmp.name, runner='setsid', guard_gib=0)
        self.profiles = Profiles(self.console)

    def test_reading_preset_does_not_apply_or_rewrite_snapshot(self):
        self.console.save_applied('demo', {'threads': '4'})
        entry = self.profiles.save('comparison', 'demo', {'threads': '8'})
        loaded = Profiles(self.console).load(entry['id'])
        self.assertEqual(loaded['options'], {'threads': '8'})
        self.assertEqual(self.console.applied_settings(), {'demo': {'threads': '4'}})
        self.assertFalse(self.console.state_file.exists())

    def test_invalid_presets_do_not_write(self):
        for name, model, values in [('', 'demo', {'threads': '4'}), ('x', 'missing', {}),
                                    ('x', 'demo', {}), ('x', 'demo', {'threads': '9'}),
                                    ('x', 'demo', {'threads': '4', 'obsolete': '1'})]:
            with self.subTest(name=name, model=model, values=values), self.assertRaises(ValueError):
                self.profiles.save(name, model, values)
        self.assertEqual(self.profiles.list(), [])

    def test_stale_preset_rejected_at_load(self):
        entry = self.profiles.save('comparison', 'demo', {'threads': '8'})
        self.console.models['demo']['options'][0]['choices'] = [('4', '4')]
        with self.assertRaises(ValueError):
            self.profiles.load(entry['id'])

    def test_apply_only_updates_after_validation_and_acceptance(self):
        self.console.save_applied('demo', {'threads': '4'})
        jobs = Jobs(self.console)
        with patch.object(self.console, 'describe', return_value='demo'), patch('threading.Thread.start'):
            with patch.object(self.console, 'build', side_effect=ValueError('incompatible')):
                with self.assertRaises(ValueError):
                    jobs.launch('demo', {'threads': '8'})
            self.assertEqual(self.console.applied_settings()['demo']['threads'], '4')
            jobs.launch('demo', {'threads': '8'})
            self.assertEqual(self.console.applied_settings()['demo']['threads'], '8')
            job, error = jobs.launch('demo', {'threads': '4'})
            self.assertIsNone(job)
            self.assertTrue(error)
            self.assertEqual(self.console.applied_settings()['demo']['threads'], '8')


if __name__ == '__main__':
    unittest.main()
