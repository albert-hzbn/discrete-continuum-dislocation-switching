"""Cell-centered Fourier initialization and finite-window coarsening diagnostics."""
import numpy as np
from scipy.signal import resample


def periodic_cell_resample(field,length,n):
    old=field.shape[0]
    if field.shape!=(old,old):raise ValueError('Require a square periodic field.')
    wave=2*np.pi*np.fft.fftfreq(old,length/old)
    shift=(length/n-length/old)/2
    shifted=np.fft.ifft2(np.fft.fft2(field)*np.exp(1j*shift*(wave[:,None]+wave[None,:]))).real
    return resample(resample(shifted,n,axis=0),n,axis=1).real


def initial_coarsening_field(length,n,seed,cutoff=1.2,rms=.04):
    base=2*int(np.ceil(length*cutoff/(2*np.pi)))+1
    if n<base:raise ValueError('Target grid does not resolve the prescribed initial spectrum.')
    rng=np.random.default_rng(seed);noise=rng.normal(size=(base,base))
    wave=2*np.pi*np.fft.fftfreq(base,length/base)
    transformed=np.fft.fft2(noise);transformed[np.hypot(wave[:,None],wave[None,:])>cutoff]=0;transformed[0,0]=0
    field=np.fft.ifft2(transformed).real;field*=rms/np.sqrt(np.mean(field**2))
    return periodic_cell_resample(field,length,n)


def coarsening_windows(time,length_s,length_e,domain):
    time=np.asarray(time);length_s=np.asarray(length_s);length_e=np.asarray(length_e);rows=[]
    for lo,hi in ((256,512),(512,1024),(1024,2048),(2048,4096)):
        selected=(time>=lo)&(time<=hi)
        if selected.sum()<3:continue
        row={'start':lo,'end':hi,'samples':int(selected.sum()),
             'max_length_over_box':float(np.max(length_s[selected])/domain),
             'min_length_over_interface_scale':float(np.min(length_s[selected])/np.sqrt(2))}
        for name,values in (('spectral',length_s),('energy',length_e)):
            x=np.log(time[selected]);y=np.log(values[selected]);coeff=np.polyfit(x,y,1)
            residual=np.sum((y-np.polyval(coeff,x))**2);variance=np.sum((y-y.mean())**2)
            row[name+'_exponent']=float(coeff[0]);row[name+'_r_squared']=float(1-residual/variance) if variance>0 else None
        row['scale_separated']=row['max_length_over_box']<.2 and row['min_length_over_interface_scale']>10
        rows.append(row)
    return rows


def radial_structure(grid,field):
    centered=field-field.mean()
    spectrum=grid.dx[0]**2*np.abs(np.fft.fft2(centered))**2/field.size
    wave=2*np.pi*np.fft.fftfreq(len(grid.x),grid.dx[0]);radius=np.hypot(wave[:,None],wave[None,:])
    spacing=2*np.pi/(grid.xe[-1]-grid.xe[0]);bins=np.rint(radius/spacing).astype(int)
    counts=np.bincount(bins.ravel());power=np.bincount(bins.ravel(),weights=spectrum.ravel())
    mean=np.divide(power,counts,out=np.zeros_like(power),where=counts>0)
    return spacing*np.arange(len(counts)),mean,counts,spectrum
