# TextBoxGan

Generate text boxes from input words.
## Setup

#### Build the docker
``` 
docker build -t textboxgan 
```

#### Run the docker
```
docker run -it -v --gpus all `pwd`:/tmp -w /tmp textboxgan bash
```

#### Download and make datasets

Inside the docker:

```
make download-and-make-datasets
```

## Network

## Implementation Details


Credits to implem de stylegan + lpips tf
