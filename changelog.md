# Changelog

This changelog includes important changes in recent updates. For a full log of all changes, please refer to the github repo.

### Version 2.1.0
* Updated FeatureClassifier, DeepRegressor, AnalyticalClassifier and GiantImpactPhaseEmulator to handle systems with 2 or fewer planets
* If system has zero or 1 planets, system is stable
* If system has two planets, models test for Hill stability (Marchal and Bozis 1982, Gladman 1993). Will return stable (infinite time to instability) if Hill stable, unstable (time to instability = 0) if not Hill stable.
* A possible pitfall with SPOCK was taking observational elements where inclinations are often close to 90 degrees. As long as input systems had low mutual inclinations (nodes appropriately chosen) there was no problem, but when sampling Omega, could inadvertently feed high mutual inclination systems into SPOCK.
* All SPOCK models now take input systems and rotate the reference axes to align with the respective system's invariable plane, and warns the user if the inclinations are outside SPOCK's training range (0.2 radians, or approximately 10 degrees)

### Version 2.0.0 
* Retrained FeatureClassifier model with cleaned Nbody dataset, now runs each system's short integration out to its fastest secular timescale rather than always 10^4 orbits (see Thadhani et al. 2025)
