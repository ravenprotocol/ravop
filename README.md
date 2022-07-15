# Ravop

Ravop is one of the crucial build blocks of Ravenverse. It is a library for requesters to create and interact with ops, perform mathematical calculations, and write algorithms. 

## Installation

```bash
pip install ravop
```

## Usage

This section covers how a requester can create a simple graph and include Ops for adding 2 Tensors using Ravop module.

>**Note:** The complete scripts of the functionalities demonstrated in this document are available in the [Ravop Repository](https://github.com/ravenprotocol/ravop) in the ```examples``` folders.

### Setting Environment Variables
Create a .env file and add the following environment variables:

```bash
RAVENVERSE_URL=http://0.0.0.0:9999
RAVENVERSE_FTP_HOST=0.0.0.0
```

Load environment variables at the beginning of your code using:

```python
from dotenv import load_dotenv
load_dotenv()
```

### Authentication and Graph Definition

The Requester must connect to the Ravenverse using a unique token that they can generate by logging in on Raven's Website using their MetaMask wallet credentials.   

```python
import ravop as R
R.initialize('<TOKEN>')
```

### Defining a Graph

In the Ravenverse, each script executed by a requester is treated as a collection of Ravop Operations called Graph.<br> 
> **Note:** In the current release, the requester can execute only 1 graph with their unique token. Therefore, to clear any previous/existing graphs, the requester must use ```R.flush()``` method. <br>

The next step involves the creation of a Graph... 

```python
R.flush()

R.Graph(name='test', algorithm='addition', approach='distributed')
```
> **Note:** ```name``` and ```algorithm``` parameters can be set to any string. However, the ```approach``` needs to be set to either "distributed" or "federated". 

### Creating Math Operations (Ops)

```python
a = R.t([1, 2, 3])
b = R.t([4, 5, 6])
c = a + b
```

### Making Ops Persist

Persisting Ops are a special category of Ops that stay in the ravenverse once the graph gets executed. The requester must explicitly mention which ops they want to save in their code. It is a good idea to write the code such that persisting ops contain the relevant results (in this case, variable - c).

> **Note:** Make sure that the ```name``` parameter for each persisting Op is unique within a graph so that later it can be retrieved.

```python
c.persist_op(name='c_output')
```

### Activate the Graph

Once all Ops of the graph have been defined, the requester must activate their graph. This step completes the compilation procedure and makes the graph ready for execution. No more Ops can be added to the graph after this.

```python
R.activate()
```

### Execute the Graph
On execution, the graph will be split into smaller subgraphs which will be distributed to the participating compute nodes in the network. The requester can also track the progress of the graph.

```python
R.execute()
R.track_progress()
```

### Fetching Results

The Requester can now fetch the computed results of the Ops that they had previously set to persist.

```python
output = R.fetch_persisting_op(op_name="c_output")
print(output)
```

## Documentation
    
[Ravop documentation](https://ravenprotocol.gitbook.io/ravenverse/ravop)


## License

<a href="https://github.com/ravenprotocol/ravop/blob/main/LICENSE.rst"><img src="https://img.shields.io/github/license/ravenprotocol/ravop"></a>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details