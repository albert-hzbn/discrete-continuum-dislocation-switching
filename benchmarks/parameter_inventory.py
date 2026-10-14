"""Record current source declarations without importing models or executing defaults."""
import argparse,ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'results/parameter-inventory')
    args=parser.parse_args()
    modules=[]
    for path in sorted((ROOT/'src/bmvr').glob('*.py')):
        source=path.read_text();tree=ast.parse(source);declarations=[]
        def visit(nodes,prefix=''):
            for node in nodes:
                if isinstance(node,ast.ClassDef):
                    fields=[{'name':x.target.id,'annotation':ast.unparse(x.annotation),
                             'default':None if x.value is None else ast.unparse(x.value)}
                            for x in node.body if isinstance(x,ast.AnnAssign) and isinstance(x.target,ast.Name)]
                    if fields:declarations.append({'kind':'class_fields','name':prefix+node.name,'line':node.lineno,'fields':fields})
                    visit(node.body,prefix+node.name+'.')
                elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                    declarations.append({'kind':'function','name':prefix+node.name,'line':node.lineno,'arguments':ast.unparse(node.args)})
        visit(tree.body)
        modules.append({'source':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(source.encode()).hexdigest(),'declarations':declarations})
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    result={'scope':'Current top-level/class source declarations; defaults are not experimental parameter values or physical-unit assignments.',
            'physical_units_registry':'PARAMETER_REGISTRY.md','modules':modules}
    (output/'api.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Recorded',len(modules),'modules and',sum(len(m['declarations']) for m in modules),'declarations.')


if __name__=='__main__':main()
