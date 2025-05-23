import yaml

#Load config file
with open('config/config.yaml', 'r') as f:
    doc = yaml.load(f)

def handler(tree,node):
    "Return config parameters"
    return doc[tree][node]
