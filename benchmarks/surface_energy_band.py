"""Section 89: propagate the surface-energy uncertainty (atomistic gamma against the liquid-extrapolated 1.75 J/m^2) into the nucleation onset."""
import json,subprocess,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results'/'embryo-atomistics-mishin-2026-09-28'/'report.json';OUT=ROOT/'results'/'surface-energy-band-2026-09-28'
GAMMA_TARGET=1.75

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);r=json.loads(SRC.read_text());f=r['fit'];s=GAMMA_TARGET/f['gamma_eff_J_m2']
    E={int(k):v for k,v in r['E_f_eV'].items()};E2={str(n):(v if n==1 else v+(s-1)*f['A']*n**(2/3)) for n,v in E.items()}
    r2=dict(r);r2['E_f_eV']=E2;r2['fit']=dict(f,A=f['A']*s,gamma_eff_J_m2=GAMMA_TARGET);r2['scope']='Section 89: Mishin embryo energies with the surface term rescaled to 1.75 J/m^2.';r2['scale']=s
    (OUT/'embryo-rescaled.json').write_text(json.dumps(r2,indent=1)+'\n')
    subprocess.run([sys.executable,str(ROOT/'benchmarks'/'nucleation_rates.py'),'--embryo',str(OUT/'embryo-rescaled.json'),'--output',str(OUT)],check=True,cwd=ROOT)
    a=json.loads((ROOT/'results'/'nucleation-rates-mishin-2026-09-28'/'report.json').read_text());b=json.loads((OUT/'report.json').read_text())
    band=dict(scale=s,onset_atomistic=a['onset_S'],onset_extrapolated=b['onset_S'],ratio={T:b['onset_S'][T]/a['onset_S'][T] for T in a['onset_S']},
        barrier_800K={str(row['S']):[x['dG_star_eV'] for x in (row,)] for row in a['table'] if row['T']==800.})
    (OUT/'band.json').write_text(json.dumps(band,indent=1)+'\n');print(json.dumps(band,indent=1))
