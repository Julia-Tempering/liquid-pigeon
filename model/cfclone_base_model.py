import numpyro
import numpy as np
import jax.numpy as jnp
from numpyro.distributions import Dirichlet, StudentT
import pandas as pd
import numpy as np
import jax.random as random  
from numpyro.infer import MCMC, NUTS

clone_cn_profiles = pd.read_csv("syns_data/clone_cn_profiles.tsv", sep="\t")
ctdna = pd.read_csv("syns_data/ctdna.tsv", sep="\t")

ctdna_cleaned = ctdna.dropna(subset=['copy'])

merged_data = pd.merge(clone_cn_profiles, ctdna_cleaned, on=["chr", "start", "end"])

clone_cn_profiles_array = merged_data[['A', 'C', 'E', 'normal']].to_numpy()
ctdna_array = merged_data['copy'].to_numpy()


def cfclone_base_model(ctdna: np.array, clone_cn_profiles: np.array, scale: float) -> None:
    """Inferred parameters are clone proportions.

    Args:
        ctdna: numpy array (n, ) of gc and mappability corrected ctdna binned read counts outputted from HMMcopy.
        clone_cn_profiles: numpy array (n, num_clones) of clone copy number profiles.
        scale: Scale (variance) parameter for student-t distribution
    """
    num_clones = clone_cn_profiles.shape[1]
    rng_key = random.PRNGKey(0)
    rho = numpyro.sample('rho', Dirichlet(jnp.ones(num_clones)), rng_key=rng_key)
    mu = jnp.log(jnp.sum(clone_cn_profiles*rho, axis=1)) - jnp.log(jnp.mean(jnp.sum(clone_cn_profiles*rho, axis=1)))
    with numpyro.plate('data', size=len(ctdna)):
        numpyro.sample('obs', StudentT(df=2, loc=mu, scale=scale), obs=ctdna)

rng_key = random.PRNGKey(0)

nuts_kernel = NUTS(cfclone_base_model)
mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=1000)
mcmc.run(rng_key, ctdna_array, clone_cn_profiles_array, 1.0)

mcmc.print_summary()

samples = mcmc.get_samples()
rho_samples = samples['rho']

print("Mean of rho:", jnp.mean(rho_samples, axis=0))