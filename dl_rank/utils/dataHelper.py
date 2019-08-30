import yaml

def writeFileToHDFS():
  rootdir = '/tmp/mnist_model'
  client = HdfsClient(hosts='localhost:50070')
  client.mkdirs('/user/root/mnist_model')
  for parent,dirnames,filenames in os.walk(rootdir):
    for dirname in  dirnames:
          print("parent is:{0}".format(parent))
    for filename in filenames:
          client.copy_from_local(os.path.join(parent,filename), os.path.join('/user/root/mnist_model',filename), overwrite=True)

def load_conf(conf_name):
    def wrapper(filename):
        with open('./{}/'.format(conf_name) + filename, 'r') as f:
            return yaml.load(f)
    return wrapper