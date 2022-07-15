from dotenv import load_dotenv
load_dotenv()

import ravop as R

R.initialize(ravenverse_token="TOKEN")

R.flush()
R.Graph(name='test', algorithm='addition', approach='distributed')

a = R.t([1, 2, 3])
b = R.t([4, 5, 6])
c = a + b

c.persist_op(name='c_output')
R.activate()

R.execute()
R.track_progress()

output = R.fetch_persisting_op(op_name="c_output")
print(output)