#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <iomanip>
using namespace std;

//Binary files → images → learn pixel patterns per digit → use probability → predict digits
//Big Endian format - Stores the most significant byte first 


//reading the data from bytes
int readData(ifstream &file){
    unsigned char bytes[4];
    //reading the 32 bit integers (file is in bytes)
    file.read((char*)bytes, 4);
    //x<<n == x*2^n (left shift operator)
    return (int(bytes[0]) << 24) |
           (int(bytes[1]) << 16) |
           (int(bytes[2]) << 8)  |
           (int(bytes[3]));
}
//filling the image data fro the file
void readDataImage(string fileName, vector<vector<unsigned char>> &imageData){
    ifstream file(fileName, ios::binary);
    if(!file){
        cout<<"Error Reading file";
        return;
    }
    int magicNumber = readData(file);
    int numOfImage = readData(file);
    int rows=readData(file);
    int cols=readData(file);

    //memory allocation for array
    imageData.resize(numOfImage, vector<unsigned char>(rows * cols));

    for (int i = 0; i < numOfImage; i++) {
        file.read((char*)imageData[i].data(), rows * cols);
    }
    file.close();
}
void readDataLabels(string filename, vector<unsigned char> &labels) {
    ifstream file(filename, ios::binary);

    if (!file) {
        cout << "Cannot open file!" << endl;
        return;
    }
    int magic = readData(file);
    int numLabels = readData(file);
    labels.resize(numLabels);
    file.read((char*)labels.data(), numLabels);
    file.close();
}
int getBinNum(unsigned char pixel){
    //each value in MNSIT is 0->256 in a 8bit grayscale we will divide it into 32 diff container
    return pixel/8;
}
void discreteMode(vector<vector<unsigned char>> &trainImageData, 
                    vector<unsigned char> &trainLabelData,        
                    vector<vector<unsigned char>> &testImageData,
                    vector<unsigned char> &testLabelData){

    vector<vector<vector<int>>> frequencyCount(10,vector<vector<int>>(784, vector<int>(32, 0)));
    vector<int> frequencyLableCount(10, 0); // frequency of number prior
    //get the pixel brightness of training data
    for(int i=0;i<trainImageData.size();i++){
        int digit = trainLabelData[i];
        frequencyLableCount[digit]++;
        for(int j=0;j<784;j++){
            //for discrete bin
            int bin = getBinNum(trainImageData[i][j]);
            frequencyCount[digit][j][bin]++;
        }
    }
    //adding pseudo count for avoiding empty bins
    for (int d = 0; d < 10; d++) {
        for (int p = 0; p < 784; p++) {
            for (int b = 0; b < 32; b++) {
            frequencyCount[d][p][b] += 1;
            }
        }
    }
    vector<double> logPost(10,0);  
    vector<double> errorDataForImage(10,0);  
//-------------------------------------------Testing--------------------------------------------------------
    for(int i=0;i<testImageData.size();i++){
        for(int d=0;d<10;d++){
        //probability of prior digit count P(D)
            logPost[d]=log(double(frequencyLableCount[d])/trainImageData.size());
            for(int j=0;j<784;j++){
                int bin = getBinNum(testImageData[i][j]); //bin of pixel in ith row of test data
                logPost[d]+=log(double(frequencyCount[d][j][bin])/(frequencyLableCount[d]+32));
            } 
        }
        double best=logPost[0]; 
        for(int k=1;k<10;k++){
            if(logPost[k]>best) best =logPost[k];
        }
        double sumExp=0;
        for(int s=0;s<logPost.size();s++)sumExp+=exp(logPost[s]-best); 
        vector<double> posterior(10);
        for(int d = 0; d < 10; d++){
            posterior[d] = exp(logPost[d]-best)/ sumExp;
        } 

        int pred =0;
        double bestVal = logPost[0];
        for(int d=0;d<10;d++){
            if(logPost[d]>bestVal){
                bestVal=logPost[d];
                pred = d;
            }
        }
        if(pred!=testLabelData[i])errorDataForImage[pred]++;
    }
//----------------------------------------Printing & Error--------------------------------------------------------
    vector<vector<double>> output(10,vector<double>(784,0));
    for(int d=0;d<10;d++){
        for(int p=0;p<784;p++){
            int bestBin=0;
            int maxFreq=0;
            for(int b=0;b<32;b++){
                if(frequencyCount[d][p][b]>maxFreq){
                    bestBin=b;
                    maxFreq=frequencyCount[d][p][b];
                }
            }
            output[d][p]=bestBin*8;
        }
    }
    //Getting the error count for each image 
    for(int i=0;i<errorDataForImage.size();i++){
        errorDataForImage[i]/=frequencyLableCount[i];
    }

    for(int d=0;d<10;d++){
        for(int p=0;p<28;p++){
           for(int c=0;c<28;c++){
               cout << (output[d][p*28 + c] > 128 ? "1" : "0");
           }
           cout << endl;
        }
        cout<<"Error % is : "<<errorDataForImage[d]<<endl;
    }
}
void conteniousGaussianMode(vector<vector<unsigned char>> &trainImageData, 
                    vector<unsigned char> &trainLabelData,        
                    vector<vector<unsigned char>> &testImageData,
                    vector<unsigned char> &testLabelData ){
    vector<vector<double>> mean(10, vector<double>(784, 0.0));
    vector<vector<double>> varience(10, vector<double>(784, 0.0));                    
    vector<int> freqCount(10,0);
//-------------------------------------------Training--------------------------------------------------------  
    //mean[3][400] = average brightness of pixel 400 in all images of digit 3
    //variance[d][j] = brightness variation of pixel J for digit d

    for(int i=0;i<trainImageData.size();i++){
        int digit = trainLabelData[i];
        freqCount[digit]++;
        for(int j=0;j<784;j++){
            mean[digit][j] +=trainImageData[i][j];
        }
    }
    //mean of each pixel wrt that digit
    for(int i=0;i<10;i++){
        for(int j=0;j<784;j++){
            mean[i][j]/=freqCount[i];
        }
    }
    //compute the vairance [Varience = pixel Value - mean]
    for(int i=0;i<trainImageData.size();i++){
        int digit = trainLabelData[i];
        for(int j=0;j<784;j++){
            double diff = trainImageData[i][j]-mean[digit][j];
            varience[digit][j] += diff*diff;
        }
    }
    for(int d=0; d<10; d++){
        for(int j=0; j<784; j++){
            varience[d][j] /= freqCount[d];
            if(varience[d][j] == 0) varience[d][j] = 0.01; // avoid zero variance
        }
    }
//-------------------------------------------Testing--------------------------------------------------------    
    vector<double> logPost(10);
    vector<double> errorDataForImages(10,0.0);
    for(int i=0;i<testImageData.size();i++){
        int lableDigit = testLabelData[i];
        for(int j=0;j<10;j++){
        logPost[j]=log(double(freqCount[j])/trainImageData.size());
            for(int k=0;k<784;k++){
                double x = testImageData[i][k];
                double m = mean[j][k];
                double s = varience[j][k];
                logPost[j] += -0.5 * log(2*M_PI*s) - ((x-m)*(x-m)) / (2*s);
            }
        }
        double best=logPost[0]; 
        int pred=0;
        for(int k=1;k<10;k++){
            if(logPost[k]>best) {
                best =logPost[k];
            }
        }
        if(pred!=testLabelData[i])errorDataForImages[lableDigit]++;

        double sumExp=0;
        for(int s=0;s<logPost.size();s++)sumExp+=exp(logPost[s]-best); 
        vector<double> posterior(10);
        for(int d = 0; d < 10; d++){
            posterior[d] = exp(logPost[d]-best)/ sumExp;
        }
        
    }
    for(int i=0;i<errorDataForImages.size();i++){
        errorDataForImages[i]/=freqCount[i];
    }
  
    //Print the values of images
    for(int d=0; d<10; d++){
    cout << "Digit " << d << " imagination:" << endl;
        for(int i=0; i<28; i++){
            for(int j=0; j<28; j++){
                cout << (mean[d][i*28 + j] > 128 ? "1" : "0");
            }
            cout << endl;
        }
        cout<<"Error % is : "<<errorDataForImages[d]<<endl;
    }
    
}
double nCrMethod(int n,int r){
    double result =0.0;
    for(int i=1;i<=r;i++){
        result = result * (n-i+1)/i;
    }
    return result;
}
void onlineLearning(){
    ifstream file("data/binarydata.txt");
    if(!file){
        cout<<"error loading the file";
    }
    
    string line;
    int caseNo=1;
    while(file>>line){
        double a,b;
        cout<<"Please input a param"<<endl;
        cin>>a;
        cout<<"Please input b param"<<endl;
        cin>>b;
        int one=0;
        int N = line.length();
        for(char c:line){
            if(c=='1')one++;
        }
        int zero = N-one;
        double probOne = double(one)/N;
        double likelihood = nCrMethod(N,one) *pow(probOne,one) *pow(1-probOne,zero);
        cout<<"case "<<caseNo++<<": "<<line<<endl;
        cout<<"Likelihood: "<<likelihood<<endl;
        cout<<"Beta prior: a = "<<a<<" b = "<<b<<endl;
        a = a + one;
        b = b + zero;
        cout<<"Beta posterior:: a = "<<a<<" b = "<<b<<endl;
    }
}

int main(int argc, char const *argv[])
{      
    cout<<fixed<<setprecision(12);
    vector<vector<unsigned char>> trainImageData; 
    vector<unsigned char> trainLabelData;        
    vector<vector<unsigned char>> testImageData; 
    vector<unsigned char> testLabelData;

    // Load training data
    readDataImage("data/train-images.idx3-ubyte__", trainImageData);
    readDataLabels("data/train-labels.idx1-ubyte__", trainLabelData);
    // Load test data
    readDataImage("data/t10k-images.idx3-ubyte__", testImageData);
    readDataLabels("data/t10k-labels.idx1-ubyte__", testLabelData);
    
    int mode;
    while(true){
        cout<<"Please enter a mode"<<"\n0.Discrete Mode    1.Contenious Mode    2.Online Learning\n";
        cin>>mode;
        if(mode==1||mode==0 || mode==2){break;}
        else{
            cout<<"Please enter a digit either 0,1,2\n";
        }
    }
    if(mode==0){
        discreteMode(trainImageData,trainLabelData,testImageData,testLabelData);
    }else if(mode ==1){
        conteniousGaussianMode(trainImageData,trainLabelData,testImageData,testLabelData);
    }else{
        onlineLearning();
    }
    return 0;
}
