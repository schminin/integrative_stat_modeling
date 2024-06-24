# TODO
* add keyword arguments for inbounds to make it clear what's happening (both parameter ("checked" by is parameter) and time (valid time)); maybe tagging parameter vector/time as Inbounds(theta) or Inbounds(t)
* precompute observation outside of the particle loop to only do ismissing check once
* compute derivatives of logpdf insides the particle loop (computing them together with the value should be faster). This would require specializing for StateSpaceFeynmanKacModels
* template_maker of additivefunctions; would make_template and using closures would be different?
* can we remove parameters completely? (we would use remake to recreate the StateSpaceModel when derivatives are to be computed)