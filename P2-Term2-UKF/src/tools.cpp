#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}
Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth) {
	  VectorXd rmse(4);
	  rmse << 0,0,0,0;
	  // Check the validity of the following inputs:
	  // The estimation vector size should not be zero
	  if(estimations.size() == 0){
	          cout << "Input is empty" << endl;
	          return rmse;
	         }
	   // The estimation vector size should equal ground truth vector size
	  if(estimations.size() != ground_truth.size()){
	           cout << "Invalid estimation or ground_truth. Data should have the same size" << endl;
	           return rmse;
	         }
	  // Accumulate squared residuals
	  for(unsigned int i=0; i < estimations.size(); ++i){
	          VectorXd residual = estimations[i] - ground_truth[i];
	          // Coefficient-wise multiplication
	          residual = residual.array()*residual.array();
	          rmse += residual;
	         }
	  // Calculate the mean
	  rmse = rmse / estimations.size();
	  rmse = rmse.array().sqrt();

	  if( rmse(0) > .08 || rmse(1) > .09 || rmse(2) > .4 || rmse(3) > .3 )
		cout << "Warning at timestep " << estimations.size() << ":  rmse = "
		<< rmse(0) << "  " << rmse(1) << "  "
		<< rmse(2) << "  " << rmse(3) << endl
		<< " currently exceeds tolerances of "
		<< ".08, .09, .4, .3" << endl;

	  return rmse;
	  }

