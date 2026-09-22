import os,tempfile,threading,time,unittest,json
from pathlib import Path
from unittest.mock import patch
from risklab.replay_runner import atomic_json

class AtomicStatusTests(unittest.TestCase):
 @unittest.skipUnless(os.name=='nt','Windows open-reader sharing semantics')
 def test_status_reader_does_not_crash_writer(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'status.json';atomic_json(p,{'version':1});opened=threading.Event()
   def reader():
    with p.open('r',encoding='utf-8') as handle:
     opened.set();time.sleep(.15);self.assertEqual(json.load(handle),{'version':1})
   thread=threading.Thread(target=reader);thread.start();opened.wait(2)
   try:atomic_json(p,{'version':2})
   finally:thread.join()
   self.assertEqual(json.loads(p.read_text()),{'version':2})
   self.assertEqual(list(Path(d).glob('*.tmp')),[])
 def test_persistent_permission_failure_is_not_silenced(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'status.json';atomic_json(p,{'version':1})
   with patch.object(Path,'replace',side_effect=PermissionError('persistent')),patch('risklab.replay_runner.time.sleep'):
    with self.assertRaises(PermissionError):atomic_json(p,{'version':2})
   self.assertEqual(json.loads(p.read_text()),{'version':1});self.assertEqual(list(Path(d).glob('*.tmp')),[])

if __name__=='__main__':unittest.main()
