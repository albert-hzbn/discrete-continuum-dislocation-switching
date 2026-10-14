"""Section 73: copper junction strengths from published DDD coefficients, and the Frank-criterion ranking test of the model."""
import argparse,json
from pathlib import Path
import numpy as np

# Kubin, Devincre, Hoc (2008), as tabulated in Akhondzadeh, Sills, Bertin, Cai, arXiv:2006.08847, Table 1.
A_IJ=dict(self=.122,coplanar=.122,Hirth=.07,Lomer=.122,glissile=.137,collinear=.625)
CU=dict(mu=54.6e9,nu=.324,b=.255e-9,drag=1.56e-5)  # copper DDD parameters of the same study
H=1/2  # a/2<110> vectors in units of a
PAIRS=dict(collinear=(np.array([1,0,1])*H,np.array([-1,0,-1])*H),Lomer=(np.array([1,0,1])*H,np.array([-1,1,0])*H),
    glissile=(np.array([1,0,1])*H,np.array([0,-1,-1])*H),Hirth=(np.array([1,0,1])*H,np.array([-1,0,1])*H))


def frank_gain(b1,b2):
    """Best of the two line senses: G = max(-2 b1.b2, 2 b1.b2), in units of a^2."""
    return float(abs(2*b1@b2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    gain={k:frank_gain(*v) for k,v in PAIRS.items()}
    junction_b={k:float(np.linalg.norm(v[0]+v[1] if frank_gain(*v)==-2*v[0]@v[1] else v[0]-v[1])) for k,v in PAIRS.items()}
    order_model=sorted(gain,key=lambda k:-gain[k]);order_ddd=sorted(gain,key=lambda k:-A_IJ[k])
    J1=bool(order_model[0]==order_ddd[0]=='collinear' and order_model[-1]==order_ddd[-1]=='Hirth' and gain['Lomer']==gain['glissile'])
    taylor={f'{rho:g}':{k:float(CU['mu']*CU['b']*np.sqrt(a*rho)/1e6) for k,a in A_IJ.items()} for rho in (1e10,1e12,1e14)}
    report=dict(scope='Section 73: copper junction strengths from published DDD coefficients; Frank-criterion ranking test (J1).',
        a_ij=A_IJ,copper_DDD_parameters=CU,frank_gain_a2=gain,junction_burgers_a=junction_b,ranking_model=order_model,ranking_DDD=order_ddd,
        gates=dict(J1_ranking=J1),taylor_MPa_by_forest_density=taylor,
        mapping='junction closure: tau_i = mu b sqrt(sum_j a_ij rho_j); the Section 54 event energy per length is kappa*G with kappa the core coefficient')
    (out/'report.json').write_text(json.dumps(report,indent=1)+'\n');print(report['gates'],gain,order_model,order_ddd)


if __name__=='__main__':main()
