function [r] = gpml_setup()
  r = addpath(genpath("gpml"));
  randn("state", [20;18]);
