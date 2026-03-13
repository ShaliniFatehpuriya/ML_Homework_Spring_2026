ofstream pred("predicted.txt");
    for(double xEle:x){
        double yEle =0;
        for(int i=0;i<weightMatrix.size();i++){
            int power =(weightMatrix.size()-1)-i;
            yEle += weightMatrix[i][0] * pow(xEle, power);
        }
    pred << xEle << " " << yEle << endl;
    }
    pred.close();
    return 0;