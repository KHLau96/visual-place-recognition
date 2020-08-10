/*//////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this
//  license. If you do not agree to this license, do not download, install,
//  copy or use the software.
//
// This file originates from the openFABMAP project:
// [http://code.google.com/p/openfabmap/] -or-
// [https://github.com/arrenglover/openfabmap]
//
// For published work which uses all or part of OpenFABMAP, please cite:
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=6224843]
//
// Original Algorithm by Mark Cummins and Paul Newman:
// [http://ijr.sagepub.com/content/27/6/647.short]
// [http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5613942]
// [http://ijr.sagepub.com/content/30/9/1100.abstract]
//
//                           License Agreement
//
// Copyright (C) 2012 Arren Glover [aj.glover@qut.edu.au] and
//                    Will Maddern [w.maddern@qut.edu.au], all rights reserved.
//
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//  * Redistribution's of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//
//  * Redistribution's in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
//  * The name of the copyright holders may not be used to endorse or promote
//    products derived from this software without specific prior written
///   permission.
//
// This software is provided by the copyright holders and contributors "as is"
// and any express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular purpose
// are disclaimed. In no event shall the Intel Corporation or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or business
// interruption) however caused and on any theory of liability, whether in
// contract, strict liability,or tort (including negligence or otherwise)
// arising in any way out of the use of this software, even if advised of the
// possibility of such damage.
//////////////////////////////////////////////////////////////////////////////*/

#include <openfabmap.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/version.hpp>

#if CV_MAJOR_VERSION == 2 and CV_MINOR_VERSION == 3

#elif CV_MAJOR_VERSION == 2 and CV_MINOR_VERSION == 4
#if USENONFREE
#include <opencv2/nonfree/nonfree.hpp>
#endif
#elif CV_MAJOR_VERSION == 3
#ifdef USENONFREE
#include <opencv2/xfeatures2d.hpp>
#endif
#endif
#include <unistd.h>

#include <fstream>
#include <iostream>
#include<string.h>
#include<stack>
#include <opencv2/opencv.hpp>
#include<cstdio>
#include <iomanip>
#include <cv.hpp>
#include "cv.h"
#define SIZE 100
/*
openFABMAP procedural functions
*/

using namespace std;
using namespace cv;

int* dist;                //dist[i]记录源顶点到i的最短距离
int* path;                //path[i]记录从源顶点到i路径上的i前面的一个顶点

struct Graph
{
    int matrix[1000][1000];  //邻接矩阵
    int vertexNum;       //顶点数
    int sideNum;         //边数
};
class Position{
public:
    void setPositionName(const string &positionName) {
        Position::positionName = positionName;
    }

    const string &getPositionName() const {
        return positionName;
    }

    int getBeginIndex() const {
        return beginIndex;
    }

    void setBeginIndex(int beginIndex) {
        Position::beginIndex = beginIndex;
    }

    int getEndIndex() const {
        return endIndex;
    }

    void setEndIndex(int endIndex) {
        Position::endIndex = endIndex;
    }

private:
    string positionName;
    int beginIndex;
    int endIndex;
protected:
};
int help(void);
int showFeatures(std::string trainPath,
                 cv::Ptr<cv::FeatureDetector> &detector);
int generateVocabTrainData(std::string trainPath,
                           std::string vocabTrainDataPath,
                           cv::Ptr<cv::FeatureDetector> &detector,
                           cv::Ptr<cv::DescriptorExtractor> &extractor);
int trainVocabulary(std::string vocabPath,
                    std::string vocabTrainDataPath,
                    double clusterRadius);

int generateBOWImageDescs(std::string dataPath,
                          std::string bowImageDescPath,
                          std::string vocabPath,
                          cv::Ptr<cv::FeatureDetector> &detector,
                          cv::Ptr<cv::DescriptorExtractor> &extractor,
                          int minWords);

int trainChowLiuTree(std::string chowliutreePath,
                     std::string fabmapTrainDataPath,
                     double lowerInformationBound);

int Sava100FramesBOWData(std::string Frames100Path,
                         std::string Frames100BowPath,
                         std::string vocabPath,
                         cv::Ptr<cv::FeatureDetector> &detector,
                         cv::Ptr<cv::DescriptorExtractor> &extractor,
                         int minWords);

Position * ShowEveryPosition(std::string EveryPositionVideoPath);

int Frames100CompareWithSingleFrames(of2::FabMap *fabmap, std::string Frames100Path,std::string Frames100BowPath,
                                     std::string ImageOnePath,
                                     std::string ImageOneBowPath,
                                     std::string Frames100CompareWithSingleFramesResultPath,
                                     std::string vocabPath,
                                     cv::Ptr<cv::FeatureDetector> &detector,
                                     cv::Ptr<cv::DescriptorExtractor> &extractor,
                                     int minWords);

int openFABMAP(std::string testPath,
               of2::FabMap *openFABMAP,
               std::string vocabPath,
               std::string resultsPath,
               bool addNewOnly);

void mergeImage(cv::Mat &dst, vector<cv::Mat> &images);
void printLocation(int number);
/*
helper functions
*/
of2::FabMap *generateFABMAPInstance(cv::FileStorage &settings);
cv::Ptr<cv::FeatureDetector> generateDetector(cv::FileStorage &fs);
cv::Ptr<cv::DescriptorExtractor> generateExtractor(cv::FileStorage &fs);

/*
Advanced tools for keypoint manipulation. These tools are not currently in the
functional code but are available for use if desired.
*/
void drawRichKeypoints(const cv::Mat& src, std::vector<cv::KeyPoint>& kpts,
                       cv::Mat& dst);
void filterKeypoints(std::vector<cv::KeyPoint>& kpts, int maxSize = 0,
                     int maxFeatures = 0);
void sortKeypoints(std::vector<cv::KeyPoint>& keypoints);

void ShowPath(Graph & graph, int & source, int v);
void Dijkstra(Graph & graph, int & source);
/*
The openFabMapcli accepts a YML settings file, an example of which is provided.
Modify options in the settings file for desired operation
*/

class GraphInital
{
public:
    GraphInital()
    {
        MaxVertex = SIZE;
        NumVertex = NumEdge = 0;//顶点数和边数初始化为0
        Vertex = new char[MaxVertex];
        Edge = new int*[MaxVertex];//int *Edge[10];
        int i,j;
        for(i = 0;i<MaxVertex;i++)
            Edge[i] = new int[MaxVertex]; //Edge[10][10]
        for(i = 0;i<MaxVertex;i++)
        {
            for(j = 0;j<MaxVertex;j++)
                Edge[i][j] = 0;
        }
    }
    void InsertVertex(char v)//插入一个顶点v
    {
        if(NumVertex >= MaxVertex)
            return;
        Vertex[NumVertex++] = v;
    }
    int GetVertexI(char v)//查找一个顶点v
    {
        int i;
        for(i = 0;i<NumVertex;i++)
        {
            if(Vertex[i] == v)
                return i;
        }
        return -1;
    }
    void InsertEdge(char v1,char v2, int distance)//插入一条由点v1和v2组成的边
    {
        int p1 = GetVertexI(v1);
        int p2 = GetVertexI(v2);
        if(p1 == -1 || p2 == -1)
            return;
        Edge[p1][p2] = Edge[p2][p1] = distance;
        NumEdge++;
    }
    void ShowGraph()//打印函数
    {
        int i,j;
        cout<<"  ";
        for(i = 0;i<NumVertex;i++)
            cout<<Vertex[i]<<" ";
        cout<<endl;
        for(i = 0;i<NumVertex;i++)
        {
            cout<<Vertex[i]<<" ";
            for(j = 0;j<NumVertex;j++)
                cout<<Edge[i][j]<<" ";
            cout<<endl;
        }
    }

    int GetEdgeNum(char v)//获取图的边数
    {
        int p = GetVertexI(v);
        if(p == -1)
            return 0;
        int n = 0;
        for(int i = 0;i<NumVertex;i++)
        {
            if(Edge[p][i] == 1)
                n++;
        }
        return n;
    }
    void DeleteVertex(char v)//删除一个顶点
    {
        int p = GetVertexI(v);
        if(p == -1)
            return;
        int i,j;
        int n=GetEdgeNum(v);
        for(i = p;i<NumVertex-1;i++)  //顶点先删除
            Vertex[i] = Vertex[i+1];

        for(i = p;i<NumVertex-1;i++)  //行上移
        {
            for(j = 0;j<NumVertex;j++)
                Edge[i][j] = Edge[i+1][j];
        }
        for(i = 0;i<NumVertex-1;i++)  //列左移
        {
            for(j = p;j<NumVertex-1;j++)
            {
                Edge[i][j] = Edge[i][j+1];
            }
        }

        NumVertex--;
        NumEdge-=n;

    }
    void DeleteEdge(char v1,char v2)//删除顶点v1和v2之间的边
    {
        int p1 = GetVertexI(v1);
        int p2 = GetVertexI(v2);
        if(p1 == -1 || p2 == -1)
            return;
        if(Edge[p1][p2] == 0)
            return;
        Edge[p1][p2] = Edge[p2][p1] = 0;
        NumEdge--;
    }
    int **getEdge(){
        return Edge;
    }

    ~GraphInital()
    {
        delete []Vertex;
        Vertex = NULL;
        for(int i=0;i<MaxVertex;i++)
        {
            delete []Edge[i];
            Edge[i] = NULL;
        }
        delete []Edge;
        Edge = NULL;
        NumVertex=0;//析构函数释放空间可不写
    }
private:
    int MaxVertex;
    int NumVertex;
    int NumEdge;
    char *Vertex;
    int **Edge;
};




int main(int argc, char * argv[])
{
    //load the settings file
    std::string settfilename;
    Position *position;
    if (argc == 1) {
        //assume settings in working directory
        settfilename = "settings.yml";
    } else if (argc == 3) {
        if(std::string(argv[1]) != "-s") {
            //incorrect option
            return help();
        } else {
            //settings provided as argument
            settfilename = std::string(argv[2]);
        }
    } else {
        //incorrect arguments
        return help();
    }

    cv::FileStorage fs;
    fs.open(settfilename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Could not open settings file: " << settfilename <<
                     std::endl;
        return -1;
    }

    cv::Ptr<cv::FeatureDetector> detector = generateDetector(fs);
    if(!detector) {
        std::cerr << "Feature Detector error" << std::endl;
        return -1;
    }

    cv::Ptr<cv::DescriptorExtractor> extractor = generateExtractor(fs);
    if(!extractor) {
        std::cerr << "Feature Extractor error" << std::endl;
        return -1;
    }

    //run desired function
    int result = 0;

    std::string function = fs["Function"];
    if (function == "ShowFeatures") {
        result = showFeatures(
                    fs["FilePaths"]["TrainPath"],
                detector);

    } else if (function == "GenerateVocabTrainData") {
        result = generateVocabTrainData(fs["FilePaths"]["TrainPath"],
                fs["FilePaths"]["TrainFeatDesc"],
                detector, extractor);

    } else if (function == "TrainVocabulary") {
        result = trainVocabulary(fs["FilePaths"]["Vocabulary"],
                fs["FilePaths"]["TrainFeatDesc"],
                fs["VocabTrainOptions"]["ClusterSize"]);

    } else if (function == "GenerateFABMAPTrainData") {
        result = generateBOWImageDescs(fs["FilePaths"]["TrainPath"],
                fs["FilePaths"]["TrainImagDesc"],
                fs["FilePaths"]["Vocabulary"], detector, extractor,
                fs["BOWOptions"]["MinWords"]);

    } else if (function == "TrainChowLiuTree") {
        result = trainChowLiuTree(fs["FilePaths"]["ChowLiuTree"],
                fs["FilePaths"]["TrainImagDesc"],
                fs["ChowLiuOptions"]["LowerInfoBound"]);

    } else if (function == "GenerateFABMAPTestData") {
        result = generateBOWImageDescs(fs["FilePaths"]["TestPath"],
                fs["FilePaths"]["TestImageDesc"],
                fs["FilePaths"]["Vocabulary"], detector, extractor,
                fs["BOWOptions"]["MinWords"]);

    } else if (function == "Sava100FramesBOWData") {
        result = Sava100FramesBOWData(fs["FilePaths"]["Frames100Path"],
                                      fs["FilePaths"]["Frames100BowPath"],
                                       fs["FilePaths"]["Vocabulary"], detector, extractor,
                                       fs["BOWOptions"]["MinWords"]);

    } else if (function == "ShowEveryPosition") {
        position = ShowEveryPosition(fs["FilePaths"]["EveryPositionVideoPath"]);

    } else if (function == "Frames100CompareWithSingleFrames") {
        of2::FabMap *fabmap = generateFABMAPInstance(fs);
        result = Frames100CompareWithSingleFrames(fabmap,fs["FilePaths"]["Frames100Path"],
                                  fs["FilePaths"]["Frames100BowPath"],
                                      fs["FilePaths"]["ImageOnePath"],
                                      fs["FilePaths"]["ImageOneBowPath"],
                                      fs["FilePaths"]["Frames100CompareWithSingleFramesResultPath"],
                                      fs["FilePaths"]["Vocabulary"], detector, extractor,
                                      fs["BOWOptions"]["MinWords"]);

    } else if (function == "RunOpenFABMAP") {
        std::string placeAddOption = fs["FabMapPlaceAddition"];
        bool addNewOnly = (placeAddOption == "NewMaximumOnly");
        of2::FabMap *fabmap = generateFABMAPInstance(fs);
        if(fabmap) {
            result = openFABMAP(fs["FilePaths"]["TestImageDesc"], fabmap,
                    fs["FilePaths"]["Vocabulary"],
                    fs["FilePaths"]["FabMapResults"], addNewOnly);
        }

    }else if(function == "Graph"){
        /////////////////////////////////////////Graph图
        //open the movie


//        if (!movie.isOpened()) {
//            std::cerr << EveryPositionVideoPath << ": training movie not found" << std::endl;
//            return nullptr;
//        }

//        GraphInital g1;
//        g1.InsertVertex('a');
//        g1.InsertVertex('b');
//        g1.InsertVertex('c');
//        g1.InsertVertex('d');
//        g1.InsertVertex('e');
//        g1.InsertVertex('f');
//        g1.InsertVertex('g');
//        g1.InsertVertex('h');
//        g1.InsertVertex('i');
//        g1.InsertVertex('j');
//        g1.InsertVertex('k');
//        g1.InsertVertex('l');
//        g1.InsertVertex('m');
//        g1.InsertVertex('n');
//        g1.InsertVertex('o');
//        g1.InsertVertex('p');
//        g1.InsertVertex('q');
//        g1.InsertVertex('r');
//        g1.InsertVertex('s');
//        g1.InsertVertex('t');
//        g1.InsertVertex('u');
//        g1.InsertVertex('v');
//        g1.InsertVertex('w');
//        g1.InsertVertex('x');
//        g1.InsertVertex('y');
//
//        g1.InsertEdge('a','b',2);
//        g1.InsertEdge('b','c',2);
//        g1.InsertEdge('c','d',2);
//        g1.InsertEdge('d','e',2);
//        g1.InsertEdge('e','f',3);
//        g1.InsertEdge('f','g',3);
//        g1.InsertEdge('g','h',3);
//        g1.InsertEdge('h','i',2);
//        g1.InsertEdge('i','j',2);
//        g1.InsertEdge('h','r',4);
//        g1.InsertEdge('j','k',2);
//        g1.InsertEdge('k','l',2);
//        g1.InsertEdge('l','m',2);
//        g1.InsertEdge('m','n',2);
//        g1.InsertEdge('n','o',2);
//        g1.InsertEdge('o','p',2);
//        g1.InsertEdge('p','q',2);
//        g1.InsertEdge('q','r',2);
//        g1.InsertEdge('r','s',2);
//        g1.InsertEdge('s','t',2);
//        g1.InsertEdge('t','u',2);
//        g1.InsertEdge('u','v',2);
//        g1.InsertEdge('v','w',2);
//        g1.InsertEdge('w','x',2);
//        g1.InsertEdge('x','y',2);
//        g1.InsertEdge('e','v',2);
//
//        g1.ShowGraph();
//
//        int ** p;
//        p = g1.getEdge();
//
//
//        Graph graph;
//        memset(graph.matrix, 0, sizeof(graph.matrix));
////    cout << "请输入图的顶点数和边数:\n";
////    cin >> graph.vertexNum >> graph.sideNum;//输入顶点数和边数
//        graph.vertexNum = 25;
//        graph.sideNum = 26;
//        dist = new int[graph.vertexNum];//内存申请
//        path = new int[graph.vertexNum];
//
////    int x, y, w;
////    cout << "请输入边的关系和权值:\n";
//        for (int i = 0; i < graph.sideNum; i++)
//        {
//            for (int j = i+1; j <graph.sideNum ; ++j) {
//                if (p[i][j] != 0){
//                    graph.matrix[i][j] = p[i][j];
//                    graph.matrix[j][i] = p[i][j];
//                }
//            }
//
//        }
//
//        cout << "\n请输入出发地点: (用0,1,2,3,4......的数字代表地点)\n ";
//        int source;
//        cin >> source;//输入源顶点
//        Dijkstra(graph, source);//求出源顶点source到其他顶点的最短路径
//        for (int i = 0; i < graph.vertexNum; i++)
//        {
//            if (i != source)
//            {
//                ShowPath(graph, source, i);//输出源顶点source到其他顶点i的最短路径
//                cout << "  The shortest path is：" << dist[i] << endl;
//            }
//        }
//
//        cout << "\n请输入要去的地点: (用0,1,2,3,4......的数字代表地点)\n ";
//        int source2;
//        cin >> source2;//输入要去的地点
//        int z = 0;
//
//
//        delete[]dist;
//        delete[]path;
    }
    else {
        std::cerr << "Incorrect Function Type" << std::endl;
        result = -1;
    }

    if (function == "Frames100CompareWithSingleFrames"){

    } else{
        std::cout << "system done" << std::endl;
        std::cin.sync(); std::cin.ignore();
    }

    if (function == "ShowEveryPosition"){
        std::cout << "Show every position done" << std::endl;

    } else{
        std::cout << "system done" << std::endl;
        std::cin.sync(); std::cin.ignore();
    }


    fs.release();
    return result;

}

/*
displays the usage message
*/
int help(void)
{
    std::cout << "Usage: openFABMAPexe -s settingsfile" << std::endl;
    return 0;
}

/*
shows the features detected on the training video
*/
int showFeatures(std::string trainPath, cv::Ptr<cv::FeatureDetector> &detector)
{

    //open the movie
    cv::VideoCapture movie;
    movie.open(trainPath);

    if (!movie.isOpened()) {
        std::cerr << trainPath << ": training movie not found" << std::endl;
        return -1;
    }

    std::cout << "Press Esc to Exit" << std::endl;
    cv::Mat frame, kptsImg;

    movie.read(frame);
    std::vector<cv::KeyPoint> kpts;
    while (movie.read(frame)) {
        detector->detect(frame, kpts);

        std::cout << kpts.size() << " keypoints detected...         \r";
        fflush(stdout);

        cv::drawKeypoints(frame, kpts, kptsImg);

        cv::imshow("Features", kptsImg);
        if(cv::waitKey(5) == 27) {
            break;
        }
    }
    std::cout << std::endl;

    cv::destroyWindow("Features");
    return 0;
}

/*
generate the data needed to train a codebook/vocabulary for bag-of-words methods
*/
int generateVocabTrainData(std::string trainPath,
                           std::string vocabTrainDataPath,
                           cv::Ptr<cv::FeatureDetector> &detector,
                           cv::Ptr<cv::DescriptorExtractor> &extractor)
{

    //Do not overwrite any files
    std::ifstream checker;
    checker.open(vocabTrainDataPath.c_str());
    if(checker.is_open()) {
        std::cerr << vocabTrainDataPath << ": Training Data already present" <<
                     std::endl;
        checker.close();
        return -1;
    }

    //load training movie
    cv::VideoCapture movie;
    movie.open(trainPath);
    if (!movie.isOpened()) {
        std::cerr << trainPath << ": training movie not found" << std::endl;
        return -1;
    }

    //extract data
    std::cout << "Extracting Descriptors" << std::endl;
    cv::Mat vocabTrainData;
    cv::Mat frame, descs, feats;
    std::vector<cv::KeyPoint> kpts;

    std::cout.setf(std::ios_base::fixed);
    std::cout.precision(0);

    while(movie.read(frame)) {

        //detect & extract features
        detector->detect(frame, kpts);
        extractor->compute(frame, kpts, descs);

        //add all descriptors to the training data
        vocabTrainData.push_back(descs);

        //show progress
        cv::drawKeypoints(frame, kpts, feats);
        cv::imshow("Training Data", feats);

        std::cout << 100.0*(movie.get(CV_CAP_PROP_POS_FRAMES) /
                            movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%. " <<
                     vocabTrainData.rows << " descriptors         \r";
        fflush(stdout);

        if(cv::waitKey(5) == 27) {
            cv::destroyWindow("Training Data");
            std::cout << std::endl;
            return -1;
        }

    }
    cv::destroyWindow("Training Data");
    std::cout << "Done: " << vocabTrainData.rows << " Descriptors" << std::endl;

    //save the training data
    cv::FileStorage fs;
    fs.open(vocabTrainDataPath, cv::FileStorage::WRITE);
    fs << "VocabTrainData" << vocabTrainData;
    fs.release();

    return 0;
}

/*
use training data to build a codebook/vocabulary
*/
int trainVocabulary(std::string vocabPath,
                    std::string vocabTrainDataPath,
                    double clusterRadius)
{

    //ensure not overwriting a vocabulary
    std::ifstream checker;
    checker.open(vocabPath.c_str());
    if(checker.is_open()) {
        std::cerr << vocabPath << ": Vocabulary already present" <<
                     std::endl;
        checker.close();
        return -1;
    }

    std::cout << "Loading vocabulary training data" << std::endl;

    cv::FileStorage fs;

    //load in vocab training data
    fs.open(vocabTrainDataPath, cv::FileStorage::READ);
    cv::Mat vocabTrainData;
    fs["VocabTrainData"] >> vocabTrainData;
    if (vocabTrainData.empty()) {
        std::cerr << vocabTrainDataPath << ": Training Data not found" <<
                     std::endl;
        return -1;
    }
    fs.release();

    std::cout << "Performing clustering" << std::endl;

    //uses Modified Sequential Clustering to train a vocabulary
    of2::BOWMSCTrainer trainer(clusterRadius);
    trainer.add(vocabTrainData);
    cv::Mat vocab = trainer.cluster();

    //save the vocabulary
    std::cout << "Saving vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::WRITE);
    fs << "Vocabulary" << vocab;
    fs.release();

    return 0;
}

/*
generate FabMap bag-of-words data : an image descriptor for each frame
*/
int generateBOWImageDescs(std::string dataPath,
                          std::string bowImageDescPath,
                          std::string vocabPath,
                          cv::Ptr<cv::FeatureDetector> &detector,
                          cv::Ptr<cv::DescriptorExtractor> &extractor,
                          int minWords)
{

    cv::FileStorage fs;

    //ensure not overwriting training data
    std::ifstream checker;
    checker.open(bowImageDescPath.c_str());
    if(checker.is_open()) {
        std::cerr << bowImageDescPath << ": FabMap Training/Testing Data "
                                         "already present" << std::endl;
        checker.close();
        return -1;
    }

    //load vocabulary
    std::cout << "Loading Vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::READ);
    cv::Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
        return -1;
    }
    fs.release();

    //use a FLANN matcher to generate bag-of-words representations
    cv::Ptr<cv::DescriptorMatcher> matcher =
            cv::DescriptorMatcher::create("FlannBased");
    cv::BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);

    //load movie
    cv::VideoCapture movie;
    movie.open(dataPath);

    if(!movie.isOpened()) {
        std::cerr << dataPath << ": movie not found" << std::endl;
        return -1;
    }

    //extract image descriptors
    cv::Mat fabmapTrainData;
    std::cout << "Extracting Bag-of-words Image Descriptors" << std::endl;
    std::cout.setf(std::ios_base::fixed);
    std::cout.precision(0);

    std::ofstream maskw;

    if(minWords) {
        maskw.open(std::string(bowImageDescPath + "mask.txt").c_str());
    }

    cv::Mat frame, bow;
    std::vector<cv::KeyPoint> kpts;

    while(movie.read(frame)) {
        detector->detect(frame, kpts);
        bide.compute(frame, kpts, bow);

        if(minWords) {
            //writing a mask file
            if(cv::countNonZero(bow) < minWords) {
                //frame masked
                maskw << "0" << std::endl;
            } else {
                //frame accepted
                maskw << "1" << std::endl;
                fabmapTrainData.push_back(bow);
            }
        } else {
            fabmapTrainData.push_back(bow);
        }

        std::cout << 100.0 * (movie.get(CV_CAP_PROP_POS_FRAMES) /
                              movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%    \r";
        fflush(stdout);
    }
    std::cout << "Done                                       " << std::endl;

    movie.release();

    //save training data
    fs.open(bowImageDescPath, cv::FileStorage::WRITE);
    fs << "BOWImageDescs" << fabmapTrainData;
    fs.release();

    return 0;
}

Position * ShowEveryPosition(std::string EveryPositionVideoPath)
{
    //open the movie

    cv::VideoCapture movie;
    movie.open(EveryPositionVideoPath);

    if (!movie.isOpened()) {
        std::cerr << EveryPositionVideoPath << ": training movie not found" << std::endl;
        return nullptr;
    }

    std::cout << "Press Esc to Exit" << std::endl;
    cv::Mat frame;

    movie.read(frame);

//    std::vector<cv::KeyPoint> kpts;
//    int z = 0;
//    while (movie.read(frame)) {
//        cv::imshow("FFFFFFF", frame);
//        if(cv::waitKey(5) == 27) {
//            break;
//        }
//        z = z+1;
//    }
//    Position obj[50];
//    int num = sizeof(obj)/ sizeof(Position);
//    for (int i = 0; i < num; ++i) {
//        obj[i].positionName = ""
//    }
    Position *obj = new Position[4];
    int num = sizeof(obj)/ sizeof(Position);
    obj[0].setPositionName("靠电梯走廊");
    obj[0].setBeginIndex(0);
    obj[0].setEndIndex(618);
    obj[1].setPositionName("靠大太阳窗户");
    obj[1].setBeginIndex(619);
    obj[1].setEndIndex(1734);
    obj[2].setPositionName("靠对侧电梯");
    obj[2].setBeginIndex(1735);
    obj[2].setEndIndex(2335);
    obj[3].setPositionName("靠医疗室");
    obj[3].setBeginIndex(2336);
    obj[3].setEndIndex(3911);

    int z = 0;
    int i = 0;
    while (movie.read(frame)) {
        if (z<=obj[i].getEndIndex()){
            cv::imshow(obj[i].getPositionName(), frame);
        } else{
            i = i+1;
        }
        if(cv::waitKey(5) == 27) {
            break;
        }
        z = z+1;
    }


//    std::cout <<"z个数" <<z <<std::endl;
//    std::cout << std::endl;

    cv::destroyWindow("靠电梯走廊");
    cv::destroyWindow("靠大太阳窗户");
    cv::destroyWindow("靠对侧电梯");
    cv::destroyWindow("靠医疗室");
    return obj;


}

int Sava100FramesBOWData(std::string Frames100Path,
                         std::string Frames100BowPath,
                         std::string vocabPath,
                         cv::Ptr<cv::FeatureDetector> &detector,
                         cv::Ptr<cv::DescriptorExtractor> &extractor,
                         int minWords)
{
    cv::FileStorage fs;

    //ensure not overwriting TwoImageResult data
    std::ifstream checker;
    checker.open(Frames100BowPath.c_str());
    if(checker.is_open()) {
        std::cerr << Frames100BowPath << ": 100 Frames Bow Result "
                                         "already present" << std::endl;
        checker.close();
        return -1;
    }

    //load vocabulary
    std::cout << "Loading Vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::READ);
    cv::Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
        return -1;
    }
    fs.release();

    //use a FLANN matcher to generate bag-of-words representations
    cv::Ptr<cv::DescriptorMatcher> matcher =
            cv::DescriptorMatcher::create("FlannBased");
    cv::BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);

    //load movie
    cv::VideoCapture movie;
    movie.open(Frames100Path);

    if(!movie.isOpened()) {
        std::cerr << Frames100Path << ": 100 frames movie not found" << std::endl;
        return -1;
    }

    //extract image descriptors from 100 frames movie
    cv::Mat frames100BowData;
    std::cout << "Extracting Bag-of-words Image Descriptors from 100 Frames Movies" << std::endl;
    std::cout.setf(std::ios_base::fixed);
    std::cout.precision(0);

    std::ofstream maskw;

    if(minWords) {
        maskw.open(std::string(Frames100BowPath + "mask.txt").c_str());
    }

    cv::Mat frame, bow;
    std::vector<cv::KeyPoint> kpts;

    while(movie.read(frame)) {
        detector->detect(frame, kpts);
        bide.compute(frame, kpts, bow);

        if(minWords) {
            //writing a mask file
            if(cv::countNonZero(bow) < minWords) {
                //frame masked
                maskw << "0" << std::endl;
            } else {
                //frame accepted
                maskw << "1" << std::endl;
                frames100BowData.push_back(bow);
            }
        } else {
            frames100BowData.push_back(bow);
        }

        std::cout << 100.0 * (movie.get(CV_CAP_PROP_POS_FRAMES) /
                              movie.get(CV_CAP_PROP_FRAME_COUNT)) << "%    \r";
        fflush(stdout);
    }
    std::cout << "Done                                       " << std::endl;

    movie.release();

    //save training data
    fs.open(Frames100BowPath, cv::FileStorage::WRITE);
    fs << "Frames100BowPath" << frames100BowData;
    fs.release();


    return 0;
}

int Frames100CompareWithSingleFrames(of2::FabMap *fabmap, std::string Frames100Path,std::string Frames100BowPath,
                                         std::string ImageOnePath,
                                         std::string ImageOneBowPath,
                                         std::string Frames100CompareWithSingleFramesResultPath,
                                         std::string vocabPath,
                                         cv::Ptr<cv::FeatureDetector> &detector,
                                         cv::Ptr<cv::DescriptorExtractor> &extractor,
                                         int minWords)
{
    char *savePath1 = "/home/samlau/CLionProjects/mycode2/matchPositions/position.txt";
    char *savePath2 = "/home/samlau/CLionProjects/mycode2/matchPositions/images2.jpg";
    char *savePath3 = "/home/samlau/CLionProjects/mycode2/Frames100CompareSingleFramesResult.yml";
    remove(savePath1);
    remove(savePath2);
    remove(savePath3);

//    if(remove(savePath1)==0 and remove(savePath2)==0 and remove(savePath3)==0)
//    {
//        cout<<"haved delete"<<endl;
//    }
//    else
//    {
//        cout<<"delete fail"<<endl;
//    }

    cv::FileStorage fs;

    //ensure not overwriting results
    std::ifstream checker;
    checker.open(Frames100CompareWithSingleFramesResultPath.c_str());
    if(checker.is_open()) {
        std::cerr << Frames100CompareWithSingleFramesResultPath << ": Results already present" << std::endl;
        checker.close();
        return -1;
    }

    //load the vocabulary
    std::cout << "Loading Vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::READ);
    cv::Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
        return -1;
    }
    fs.release();

    //load the test data
    fs.open(Frames100BowPath, cv::FileStorage::READ);
    cv::Mat testImageDescs;
    fs["Frames100BowPath"] >> testImageDescs;
    if(testImageDescs.empty()) {
        std::cerr << Frames100BowPath << ": Test data not found" << std::endl;
        return -1;
    }
    fs.release();

    //use a FLANN matcher to generate bag-of-words representations
    cv::Ptr<cv::DescriptorMatcher> matcher =
            cv::DescriptorMatcher::create("FlannBased");
    cv::BOWImgDescriptorExtractor bide(extractor, matcher);
    bide.setVocabulary(vocab);

    //load image1
    cv::Mat image1 = cv::imread(ImageOnePath);
    if(image1.empty()){
        std::cerr << ImageOnePath << ": image 1 not found" << std::endl;
        return -1;
    }

    cv::Mat bow, image1BowData;
    std::vector<cv::KeyPoint> kpts;
    detector->detect(image1, kpts);
    bide.compute(image1, kpts, bow);
    image1BowData.push_back(bow);

    //save image1 Bow data
    fs.open(ImageOneBowPath, cv::FileStorage::WRITE);
    fs << "ImageOneBowPath" << image1BowData;
    fs.release();

    //image1 Bow + frames 100 Bow
    cv::Mat totalBowdata;
    totalBowdata.push_back(testImageDescs);
    totalBowdata.push_back(image1BowData);

    //running COMPARISION
    std::cout << "Running COMPARISION" << std::endl;
    std::vector<of2::IMatch> matches;
    std::vector<of2::IMatch>::iterator l;
    cv::Mat confusion_mat(totalBowdata.rows, totalBowdata.rows, CV_64FC1);
    confusion_mat.setTo(0); // init to 0's

    fabmap->localize(totalBowdata,matches,true);
    std::cout << matches.size() << std::endl;
    for(l = matches.begin(); l != matches.end(); l++) {
        if(l->imgIdx < 0) {
            confusion_mat.at<double>(l->queryIdx, l->queryIdx) = l->match;

        } else {
            confusion_mat.at<double>(l->queryIdx, l->imgIdx) = l->match;
        }
    }

    //save the result as plain text for ease of import to Matlab
    std::ofstream writer(Frames100CompareWithSingleFramesResultPath.c_str());
//    for(int i = 0; i < confusion_mat.rows; i++) {
    vector<int> pmark;
    for(int i = confusion_mat.rows-1; i < confusion_mat.rows; i++) {
        for(int j = 0; j < confusion_mat.cols; j++) {
            writer << confusion_mat.at<double>(i, j) << " ";
            if (confusion_mat.at<double>(i, j)>=0.5 and confusion_mat.at<double>(i, j)<1){
                pmark.push_back(j);
            }
        }
        writer << std::endl;
    }
    writer.close();
    for (int k = 0; k <pmark.size() ; ++k) {
        cout << "pmark " << pmark[k] << endl;
    }

    ofstream ofile;
    ofile.open("../matchPositions/position.txt");
    ofile << pmark[0] << endl;
    ofile.close();

    cv::VideoCapture movie;
    vector<cv::Mat> images;
    cv::Mat dst;
    movie.open(Frames100Path);
    cv::Mat frame;
    int pframe = 0;
    int pmark_index = 0;
    images.push_back(image1);

    while(movie.read(frame)) {
        if(pframe == pmark[pmark_index]){
            cv::imwrite("/home/samlau/CLionProjects/mycode2/matchPositions/image2.jpg",frame);
            pmark_index++;
        }

        pframe++;
    }
    Mat image2 = cv::imread("/home/samlau/CLionProjects/mycode2/matchPositions/image2.jpg");
    images.push_back(image2);
    mergeImage(dst,images);



    GraphInital g1;
    g1.InsertVertex('a');
    g1.InsertVertex('b');
    g1.InsertVertex('c');
    g1.InsertVertex('d');
    g1.InsertVertex('e');
    g1.InsertVertex('f');
    g1.InsertVertex('g');
    g1.InsertVertex('h');
    g1.InsertVertex('i');
    g1.InsertVertex('j');
    g1.InsertVertex('k');
    g1.InsertVertex('l');
    g1.InsertVertex('m');
    g1.InsertVertex('n');
    g1.InsertVertex('o');
    g1.InsertVertex('p');
    g1.InsertVertex('q');
    g1.InsertVertex('r');
    g1.InsertVertex('s');
    g1.InsertVertex('t');
    g1.InsertVertex('u');
    g1.InsertVertex('v');
    g1.InsertVertex('w');
    g1.InsertVertex('x');
    g1.InsertVertex('y');

    g1.InsertEdge('a','b',2);
    g1.InsertEdge('b','c',2);
    g1.InsertEdge('c','d',2);
    g1.InsertEdge('d','e',2);
    g1.InsertEdge('e','f',3);
    g1.InsertEdge('f','g',3);
    g1.InsertEdge('g','h',3);
    g1.InsertEdge('h','i',2);
    g1.InsertEdge('i','j',2);
    g1.InsertEdge('h','r',4);
    g1.InsertEdge('j','k',2);
    g1.InsertEdge('k','l',2);
    g1.InsertEdge('l','m',2);
    g1.InsertEdge('m','n',2);
    g1.InsertEdge('n','o',2);
    g1.InsertEdge('o','p',2);
    g1.InsertEdge('p','q',2);
    g1.InsertEdge('q','r',2);
    g1.InsertEdge('r','s',2);
    g1.InsertEdge('s','t',2);
    g1.InsertEdge('t','u',2);
    g1.InsertEdge('u','v',2);
    g1.InsertEdge('v','w',2);
    g1.InsertEdge('w','x',2);
    g1.InsertEdge('x','y',2);
    g1.InsertEdge('e','v',2);
    int ** p;
    p = g1.getEdge();

    Graph graph;
    memset(graph.matrix, 0, sizeof(graph.matrix));
//    cout << "请输入图的顶点数和边数:\n";
//    cin >> graph.vertexNum >> graph.sideNum;//输入顶点数和边数
    graph.vertexNum = 25;
    graph.sideNum = 26;
    dist = new int[graph.vertexNum];//内存申请
    path = new int[graph.vertexNum];
    for (int i = 0; i < graph.sideNum; i++)
    {
        for (int j = i+1; j <graph.sideNum ; ++j) {
            if (p[i][j] != 0){
                graph.matrix[i][j] = p[i][j];
                graph.matrix[j][i] = p[i][j];
            }
        }
    }
//53-798 haidilao
//799-1234 zhuangxiu_1
//1235- 1657walaile
//1658-1941 gongan
//1942-2524 wanjufandoucheng
//-2823 aiyingdao
//-3102    50lan
//-3581    fila
//-3738   aimer
//-3902   patco
//-4116   annil
//-4279   aiyingdao
// -4402 xiaoxaingQB + meilianyingyu
// -4801 壁画tough
// -5239 qinhang
//  -5708 antakid
// - 5911 dr kong
// -6192 mumu mianpinjia
// - 6647 zhuangxiu 2
// - 6914   2pingmi
// -7152 kid land
// -7439  lego
// -7795 balabala
// - 8172 hongsechuchuang
// -8489 muyingyongpin
// -8722 zhuangxiu_3
    cout << ""<<endl;
    int source;
    if (pmark[0] >52 and pmark[0]<=798){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'HaiDiLao\' "<<endl;
        source = 0;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >798 and pmark[0]<=1234){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Fitment Area 1\' "<<endl;
        source = 0;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >1234 and pmark[0]<=1657){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'WaLaile\' "<<endl;
        source = 2;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >1657 and pmark[0]<=1941){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Public Security Bureau Self-Service System\' "<<endl;
        source = 3;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >1941 and pmark[0]<=2524){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Toys R Us\' "<<endl;
        source = 4;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >2524 and pmark[0]<=2823){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'AiYingDao No.1\' "<<endl;
        source = 5;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >2823 and pmark[0]<=3102){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'50 Lan\' "<<endl;
        source = 6;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >3102 and pmark[0]<=3581){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Fila\' "<<endl;
        source = 7;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >3581 and pmark[0]<=3738){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'aimer\' "<<endl;
        source = 8;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >3738 and pmark[0]<=3902){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'patco\' "<<endl;
        source = 9;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >3902 and pmark[0]<=4116){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'annil\' "<<endl;
        source = 10;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >4116 and pmark[0]<=4279){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'AiYingDao No.2\' "<<endl;
        source = 11;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >4279 and pmark[0]<=4402){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'XIAOXIANGQB and MeiLian English\' "<<endl;
        source = 12;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >4402 and pmark[0]<=4801){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'TOUGH\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >4801 and pmark[0]<=5239){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'PIANO MALL\' "<<endl;
        source = 13;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >5239 and pmark[0]<=5708){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'ANTA Kid\' "<<endl;
        source = 14;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >5708 and pmark[0]<=5911){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Dr Kong\' "<<endl;
        source = 15;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >5911 and pmark[0]<=6192){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'MuMu button clothes\' "<<endl;
        source = 16;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >6192 and pmark[0]<=6647){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Fitment Area 2\' "<<endl;
        source = 17;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >6647 and pmark[0]<=6914){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'2 Pingmi\' "<<endl;
        source = 18;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >6914 and pmark[0]<=7152){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Kid Land\' "<<endl;
        source = 19;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >7152 and pmark[0]<=7439){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Lego\' "<<endl;
        source = 20;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >7439 and pmark[0]<=7795){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Balabala\' "<<endl;
        source = 21;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >7795 and pmark[0]<=8172){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Red Shwocase\' "<<endl;
        source = 22;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >8172 and pmark[0]<=8489){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Mother & baby shop\' "<<endl;
        source = 23;
        cout << "########################################################################" <<endl;
    }
    else if (pmark[0] >8489 and pmark[0]<=8838){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Fitment Area 3\' "<<endl;
        source = 24;
        cout << "########################################################################" <<endl;
    }
    cout << ""<<endl;


    Dijkstra(graph, source);//求出源顶点source到其他顶点的最短路径
    for (int i = 0; i < graph.vertexNum; i++)
    {
        if (i != source)
        {
            ShowPath(graph, source, i);//输出源顶点source到其他顶点i的最短路径
            cout << "  The shortest path is:" << dist[i] << endl;
        }
    }

    string flaggg = "YES";
    string numberstring = "";
    while (flaggg == "YES"){
        cout << "Where do you want to go?"<<endl;
        cout << "[0].HaiDiLao        [1].Fitment Area1      [2].WaLaiLe        [3].Public Security Bureau Self-Service System" <<endl;
        cout << "[4].Toys R Us       [5].AiYingDao No.1     [6].50 Lan         [7].Fila" <<endl;
        cout << "[8].Aimer&Anta      [9].Patco&Annil        [10].AiYingDao No.2[11].XIAOXIANGQB and MeiLian English" <<endl;
        cout << "[12].TOUGH          [13].PIANO MALL        [14].ANTA Kid      [15].Dr Kong" <<endl;
        cout << "[16].MuMu clothes   [17].Fitment Area2     [18].2Pingmi       [19].Kid Land" <<endl;
        cout << "[20].Lego           [21].Balabala          [22].Red Shwocase  [23].Mother & baby shop" <<endl;
        cout << "[24].Fitment Area3" <<endl;
        cout << "Please enter your destination: "<<endl;

        int number = 0;
        cin >> number;

        if (numberstring == "HaiDiLao"){
            number = 0;
        }
        if (numberstring == "Fitment Area1"){
            number = 1;
        }
        if (numberstring == "WaLaiLe"){
            number = 2;
        }
        if (numberstring == "Public Security Bureau Self-Service System"){
            number = 3;
        }
        if (numberstring == "Toys R Us"){
            number = 4;
        }
        if (numberstring == "AiYingDao No.1"){
            number = 5;
        }
        if (numberstring == "50 Lan"){
            number = 6;
        }
        if (numberstring == "Fila"){
            number = 7;
        }
        if (numberstring == "Aimer&Anta"){
            number = 8;
        }
        if (numberstring == "Patco&Annil"){
            number = 9;
        }
        if (numberstring == "AiYingDao No.2"){
            number = 10;
        }
        if (numberstring == "XIAOXIANGQB and MeiLian English"){
            number = 11;
        }
        if (numberstring == "TOUGH"){
            number = 12;
        }
        if (numberstring == "PIANO MALL"){
            number = 13;
        }
        if (numberstring == "ANTA Kid"){
            number = 14;
        }
        if (numberstring == "Dr Kong"){
            number = 15;
        }
        if (numberstring == "MuMu clothes"){
            number = 16;
        }
        if (numberstring == "Fitment Area2"){
            number = 17;
        }
        if (numberstring == "2Pingmi"){
            number = 18;
        }
        if (numberstring == "Kid Land"){
            number = 19;
        }
        if (numberstring == "Lego"){
            number = 20;
        }
        if (numberstring == "Balabala"){
            number = 21;
        }
        if (numberstring == "Red Shwocase"){
            number = 22;
        }
        if (numberstring == "Mother & baby shop"){
            number = 23;
        }
        if (numberstring == "Fitment Area3"){
            number = 24;
        }

        if (number != -1){
            cout << "Planning your route..."<<endl;
            VideoCapture capture;
            string videoplayingPath = "/home/samlau/CLionProjects/mycode2/pathNavigVideo/";
            videoplayingPath = videoplayingPath + std::to_string(source)+"-"+std::to_string(number) +".mp4";
            capture.open(videoplayingPath);

            while (1)
            {
                Mat frame;
                capture >> frame;
                if (frame.empty())
                {
                    break;
                }
//                putText(frame,"hello",Point(160,480),FONT_HERSHEY_SIMPLEX,4,Scalar(255,0,0),4,8);
                imshow("Navigation", frame);
                waitKey(30);  //延时30ms
            }

            if (number == 5){
                Mat img1 = imread("/home/samlau/CLionProjects/mycode2/50.jpg",1);
                putText(img1,"ARRIVAL",Point(160,400),FONT_HERSHEY_SIMPLEX,3,Scalar(255,0,0),4,8);
                imshow("Navigation",img1);
            }

            waitKey(5000);


            cv::destroyAllWindows();

            //        cv::destroyWindow("Navigation");
            printLocation(number);


        }
        cout <<"Do you want to go somewhere else from here?(YES/NO)" << endl;
        cin >> flaggg;
        numberstring = "";
        source = number;
    }

    return 0;
}

void printLocation(int number){
    if (number == 0){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'HaiDiLao\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 1){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Fitment Area 1\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 2){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'WaLaile\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 3){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Public Security Bureau Self-Service System\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 4){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Toys R Us\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 5){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'AiYingDao No.1\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 6){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'50 Lan\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 7){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Fila\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 8){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'aimer\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 9){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Patco&Annil\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 10){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'AiYingDao No.2\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 11){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'XIAOXIANGQB and MeiLian English\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 12){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'TOUGH\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 13){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'PIANO MALL\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 14){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'ANTA Kid\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 15){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Dr Kong\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 16){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'MuMu button clothes\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 17){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Fitment Area 2\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 18){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'2 Pingmi\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 19){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Kid Land\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 20){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Lego\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 21){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Balabala\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 22){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Red Shwocase\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 23){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Mother & baby shop\' "<<endl;
        cout << "########################################################################" <<endl;
    }
    else if (number == 24){
        cout << "########################################################################" <<endl;
        cout << "You are in front of \'Fitment Area 3\' "<<endl;
        cout << "########################################################################" <<endl;
    }
}

void mergeImage(cv::Mat &dst, vector<cv::Mat> &images)
{
    int imgCount = (int)images.size();

    if (imgCount <= 0)
    {
        printf("the number of images is too small\n");
        return;
    }



    /*将每个图片缩小为指定大小*/
    int rows = 300;
    int cols = 400;
    for (int i = 0; i < imgCount; i++)
    {
        resize(images[i], images[i], Size(cols, rows)); //注意区别：Size函数的两个参数分别为：宽和高，宽对应cols，高对应rows
    }

    /*创建新图片的尺寸
        高：rows * imgCount/2
        宽：cols * 2
    */
    dst.create(rows * imgCount / 2, cols * 2, CV_8UC3);

    for (int i = 0; i < imgCount; i++)
    {
        images[i].copyTo(dst(Rect((i % 2) * cols, (i / 2)*rows, images[0].cols, images[0].rows)));
    }
}
/*
generate a Chow-Liu tree from FabMap Training data
*/
int trainChowLiuTree(std::string chowliutreePath,
                     std::string fabmapTrainDataPath,
                     double lowerInformationBound)
{

    cv::FileStorage fs;

    //ensure not overwriting training data
    std::ifstream checker;
    checker.open(chowliutreePath.c_str());
    if(checker.is_open()) {
        std::cerr << chowliutreePath << ": Chow-Liu Tree already present" <<
                     std::endl;
        checker.close();
        return -1;
    }

    //load FabMap training data
    std::cout << "Loading FabMap Training Data" << std::endl;
    fs.open(fabmapTrainDataPath, cv::FileStorage::READ);
    cv::Mat fabmapTrainData;
    fs["BOWImageDescs"] >> fabmapTrainData;
    if (fabmapTrainData.empty()) {
        std::cerr << fabmapTrainDataPath << ": FabMap Training Data not found"
                  << std::endl;
        return -1;
    }
    fs.release();

    //generate the tree from the data
    std::cout << "Making Chow-Liu Tree" << std::endl;
    of2::ChowLiuTree tree;
    tree.add(fabmapTrainData);
    cv::Mat clTree = tree.make(lowerInformationBound);

    //save the resulting tree
    std::cout <<"Saving Chow-Liu Tree" << std::endl;
    fs.open(chowliutreePath, cv::FileStorage::WRITE);
    fs << "ChowLiuTree" << clTree;
    fs.release();

    return 0;

}


/*
Run FabMap on a test dataset
*/
int openFABMAP(std::string testPath,
               of2::FabMap *fabmap,
               std::string vocabPath,
               std::string resultsPath,
               bool addNewOnly)
{

    cv::FileStorage fs;

    //ensure not overwriting results
    std::ifstream checker;
    checker.open(resultsPath.c_str());
    if(checker.is_open()) {
        std::cerr << resultsPath << ": Results already present" << std::endl;
        checker.close();
        return -1;
    }

    //load the vocabulary
    std::cout << "Loading Vocabulary" << std::endl;
    fs.open(vocabPath, cv::FileStorage::READ);
    cv::Mat vocab;
    fs["Vocabulary"] >> vocab;
    if (vocab.empty()) {
        std::cerr << vocabPath << ": Vocabulary not found" << std::endl;
        return -1;
    }
    fs.release();

    //load the test data
    fs.open(testPath, cv::FileStorage::READ);
    cv::Mat testImageDescs;
    fs["BOWImageDescs"] >> testImageDescs;
    if(testImageDescs.empty()) {
        std::cerr << testPath << ": Test data not found" << std::endl;
        return -1;
    }
    fs.release();

    //running openFABMAP
    std::cout << "Running openFABMAP" << std::endl;
    std::vector<of2::IMatch> matches;
    std::vector<of2::IMatch>::iterator l;



    cv::Mat confusion_mat(testImageDescs.rows, testImageDescs.rows, CV_64FC1);
    confusion_mat.setTo(0); // init to 0's


    if (!addNewOnly) {

        //automatically comparing a whole dataset
        fabmap->localize(testImageDescs, matches, true);
        std::cout << matches.size() << std::endl;
        for(l = matches.begin(); l != matches.end(); l++) {
            if(l->imgIdx < 0) {
                confusion_mat.at<double>(l->queryIdx, l->queryIdx) = l->match;

            } else {
                confusion_mat.at<double>(l->queryIdx, l->imgIdx) = l->match;
            }
        }

    } else {

        //criteria for adding locations used
        for(int i = 0; i < testImageDescs.rows; i++) {
            matches.clear();
            //compare images individually
            fabmap->localize(testImageDescs.row(i), matches);

            bool new_place_max = true;
            for(l = matches.begin(); l != matches.end(); l++) {

                if(l->imgIdx < 0) {
                    //add the new place to the confusion matrix 'diagonal'
                    confusion_mat.at<double>(i, (int)matches.size()-1) = l->match;

                } else {
                    //add the score to the confusion matrix
                    confusion_mat.at<double>(i, l->imgIdx) = l->match;
                }

                //test for new location maximum
                if(l->match > matches.front().match) {
                    new_place_max = false;
                }
            }

            if(new_place_max) {
                fabmap->add(testImageDescs.row(i));
            }
        }
    }

    //save the result as plain text for ease of import to Matlab
    std::ofstream writer(resultsPath.c_str());
    for(int i = 0; i < confusion_mat.rows; i++) {
        for(int j = 0; j < confusion_mat.cols; j++) {
            writer << confusion_mat.at<double>(i, j) << " ";
        }
        writer << std::endl;
    }
    writer.close();

    return 0;
}

#if CV_MAJOR_VERSION == 2
/*
generates a feature detector based on options in the settings file
*/
cv::Ptr<cv::FeatureDetector> generateDetector(cv::FileStorage &fs) {

    //create common feature detector and descriptor extractor
    std::string detectorMode = fs["FeatureOptions"]["DetectorMode"];
    std::string detectorType = fs["FeatureOptions"]["DetectorType"];
    cv::Ptr<cv::FeatureDetector> detector = NULL;
    if(detectorMode == "ADAPTIVE") {

        if(detectorType != "STAR" && detectorType != "SURF" &&
                detectorType != "FAST") {
            std::cerr << "Adaptive Detectors only work with STAR, SURF "
                         "and FAST" << std::endl;
        } else {

            detector = new cv::DynamicAdaptedFeatureDetector(
                        cv::AdjusterAdapter::create(detectorType),
                        fs["FeatureOptions"]["Adaptive"]["MinFeatures"],
                    fs["FeatureOptions"]["Adaptive"]["MaxFeatures"],
                    fs["FeatureOptions"]["Adaptive"]["MaxIters"]);
        }

    } else if(detectorMode == "STATIC") {
        if(detectorType == "STAR") {

            detector = new cv::StarFeatureDetector(
                        fs["FeatureOptions"]["StarDetector"]["MaxSize"],
                    fs["FeatureOptions"]["StarDetector"]["Response"],
                    fs["FeatureOptions"]["StarDetector"]["LineThreshold"],
                    fs["FeatureOptions"]["StarDetector"]["LineBinarized"],
                    fs["FeatureOptions"]["StarDetector"]["Suppression"]);

        } else if(detectorType == "FAST") {

            detector = new cv::FastFeatureDetector(
                        fs["FeatureOptions"]["FastDetector"]["Threshold"],
                    (int)fs["FeatureOptions"]["FastDetector"]
                    ["NonMaxSuppression"] > 0);
        } else if(detectorType == "MSER") {

            detector = new cv::MserFeatureDetector(
                        fs["FeatureOptions"]["MSERDetector"]["Delta"],
                    fs["FeatureOptions"]["MSERDetector"]["MinArea"],
                    fs["FeatureOptions"]["MSERDetector"]["MaxArea"],
                    fs["FeatureOptions"]["MSERDetector"]["MaxVariation"],
                    fs["FeatureOptions"]["MSERDetector"]["MinDiversity"],
                    fs["FeatureOptions"]["MSERDetector"]["MaxEvolution"],
                    fs["FeatureOptions"]["MSERDetector"]["AreaThreshold"],
                    fs["FeatureOptions"]["MSERDetector"]["MinMargin"],
                    fs["FeatureOptions"]["MSERDetector"]["EdgeBlurSize"]);
#if USENONFREE
        } else if(detectorType == "SURF") {

#if CV_MINOR_VERSION == 4
            detector = new cv::SURF(
                        fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                    (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                    (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

#elif CV_MINOR_VERSION == 3
            detector = new cv::SurfFeatureDetector(
                        fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                    (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
#endif
        } else if(detectorType == "SIFT") {
#if CV_MINOR_VERSION == 4
            detector = new cv::SIFT(
                        fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                    fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                    fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                    fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                    fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#elif CV_MINOR_VERSION == 3
            detector = new cv::SiftFeatureDetector(
                        fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                    fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"]);
#endif
#endif //USENONFREE

        } else {
            std::cerr << "Could not create detector class. Specify detector "
                         "options in the settings file" << std::endl;
        }
    } else {
        std::cerr << "Could not create detector class. Specify detector "
                     "mode (static/adaptive) in the settings file" << std::endl;
    }

    return detector;

}
#elif CV_MAJOR_VERSION == 3
/*
generates a feature detector based on options in the settings file
*/
cv::Ptr<cv::FeatureDetector> generateDetector(cv::FileStorage &fs) {

    //create common feature detector and descriptor extractor
    std::string detectorType = fs["FeatureOptions"]["DetectorType"];

    if(detectorType == "BRISK") {
        return cv::BRISK::create(
                    fs["FeatureOptions"]["BRISK"]["Threshold"],
                fs["FeatureOptions"]["BRISK"]["Octaves"],
                fs["FeatureOptions"]["BRISK"]["PatternScale"]);
    } else if(detectorType == "ORB") {
        return cv::ORB::create(
                    fs["FeatureOptions"]["ORB"]["nFeatures"],
                fs["FeatureOptions"]["ORB"]["scaleFactor"],
                fs["FeatureOptions"]["ORB"]["nLevels"],
                fs["FeatureOptions"]["ORB"]["edgeThreshold"],
                fs["FeatureOptions"]["ORB"]["firstLevel"],
                2, cv::ORB::HARRIS_SCORE,
                fs["FeatureOptions"]["ORB"]["patchSize"]);

    } else if(detectorType == "MSER") {
        return cv::MSER::create(
                    fs["FeatureOptions"]["MSERDetector"]["Delta"],
                fs["FeatureOptions"]["MSERDetector"]["MinArea"],
                fs["FeatureOptions"]["MSERDetector"]["MaxArea"],
                fs["FeatureOptions"]["MSERDetector"]["MaxVariation"],
                fs["FeatureOptions"]["MSERDetector"]["MinDiversity"],
                fs["FeatureOptions"]["MSERDetector"]["MaxEvolution"],
                fs["FeatureOptions"]["MSERDetector"]["AreaThreshold"],
                fs["FeatureOptions"]["MSERDetector"]["MinMargin"],
                fs["FeatureOptions"]["MSERDetector"]["EdgeBlurSize"]);
    } else if(detectorType == "FAST") {
        return cv::FastFeatureDetector::create(
                    fs["FeatureOptions"]["FastDetector"]["Threshold"],
                (int)fs["FeatureOptions"]["FastDetector"]["NonMaxSuppression"] > 0);
    } else if(detectorType == "AGAST") {
        return cv::AgastFeatureDetector::create(
                    fs["FeatureOptions"]["AGAST"]["Threshold"],
                (int)fs["FeatureOptions"]["AGAST"]["NonMaxSuppression"] > 0);
#ifdef USENONFREE
    } else if(detectorType == "STAR") {

        return cv::xfeatures2d::StarDetector::create(
                    fs["FeatureOptions"]["StarDetector"]["MaxSize"],
                fs["FeatureOptions"]["StarDetector"]["Response"],
                fs["FeatureOptions"]["StarDetector"]["LineThreshold"],
                fs["FeatureOptions"]["StarDetector"]["LineBinarized"],
                fs["FeatureOptions"]["StarDetector"]["Suppression"]);

    } else if(detectorType == "SURF") {
        return cv::xfeatures2d::SURF::create(
                    fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

    } else if(detectorType == "SIFT") {
        return cv::xfeatures2d::SIFT::create(
                    fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#endif

    } else {
        std::cerr << "Could not create detector class. Specify detector "
                     "mode (static/adaptive) in the settings file" << std::endl;
    }

    return cv::Ptr<cv::FeatureDetector>(); //return the nullptr

}
#endif


/*
generates a feature detector based on options in the settings file
*/
#if CV_MAJOR_VERSION == 2
cv::Ptr<cv::DescriptorExtractor> generateExtractor(cv::FileStorage &fs)
{
    std::string extractorType = fs["FeatureOptions"]["ExtractorType"];
    cv::Ptr<cv::DescriptorExtractor> extractor = NULL;
    if(extractorType == "DUMMY") {

#ifdef USENONFREE
    } else if(extractorType == "SIFT") {
#if CV_MINOR_VERSION == 4
        extractor = new cv::SIFT(
                    fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#elif CV_MINOR_VERSION == 3
        extractor = new cv::SiftDescriptorExtractor();
#endif

    } else if(extractorType == "SURF") {

#if CV_MINOR_VERSION == 4
        extractor = new cv::SURF(
                    fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

#elif CV_MINOR_VERSION == 3
        extractor = new cv::SurfDescriptorExtractor(
                    fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);
#endif
#endif

    } else {
        std::cerr << "Could not create Descriptor Extractor. Please specify "
                     "extractor type in settings file" << std::endl;
    }

    return extractor;

}
#elif CV_MAJOR_VERSION == 3

cv::Ptr<cv::DescriptorExtractor> generateExtractor(cv::FileStorage &fs)
{
    std::string extractorType = fs["FeatureOptions"]["ExtractorType"];

    if(extractorType == "BRISK") {
        return cv::BRISK::create(
                    fs["FeatureOptions"]["BRISK"]["Threshold"],
                fs["FeatureOptions"]["BRISK"]["Octaves"],
                fs["FeatureOptions"]["BRISK"]["PatternScale"]);
    } else if(extractorType == "ORB") {
        return cv::ORB::create(
                    fs["FeatureOptions"]["ORB"]["nFeatures"],
                fs["FeatureOptions"]["ORB"]["scaleFactor"],
                fs["FeatureOptions"]["ORB"]["nLevels"],
                fs["FeatureOptions"]["ORB"]["edgeThreshold"],
                fs["FeatureOptions"]["ORB"]["firstLevel"],
                2, cv::ORB::HARRIS_SCORE,
                fs["FeatureOptions"]["ORB"]["patchSize"]);
#ifdef USENONFREE
    } else if(extractorType == "SURF") {
        return cv::xfeatures2d::SURF::create(
                    fs["FeatureOptions"]["SurfDetector"]["HessianThreshold"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaves"],
                fs["FeatureOptions"]["SurfDetector"]["NumOctaveLayers"],
                (int)fs["FeatureOptions"]["SurfDetector"]["Extended"] > 0,
                (int)fs["FeatureOptions"]["SurfDetector"]["Upright"] > 0);

    } else if(extractorType == "SIFT") {
        return cv::xfeatures2d::SIFT::create(
                    fs["FeatureOptions"]["SiftDetector"]["NumFeatures"],
                fs["FeatureOptions"]["SiftDetector"]["NumOctaveLayers"],
                fs["FeatureOptions"]["SiftDetector"]["ContrastThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["EdgeThreshold"],
                fs["FeatureOptions"]["SiftDetector"]["Sigma"]);
#endif

    } else {
        std::cerr << "Could not create Descriptor Extractor. Please specify "
                     "extractor type in settings file" << std::endl;
    }

    return cv::Ptr<cv::DescriptorExtractor>();

}

#endif



/*
create an instance of a FabMap class with the options given in the settings file
*/
of2::FabMap *generateFABMAPInstance(cv::FileStorage &settings)
{

    cv::FileStorage fs;

    //load FabMap training data
    std::string fabmapTrainDataPath = settings["FilePaths"]["TrainImagDesc"];
    std::string chowliutreePath = settings["FilePaths"]["ChowLiuTree"];

    std::cout << "Loading FabMap Training Data" << std::endl;
    fs.open(fabmapTrainDataPath, cv::FileStorage::READ);
    cv::Mat fabmapTrainData;
    fs["BOWImageDescs"] >> fabmapTrainData;
    if (fabmapTrainData.empty()) {
        std::cerr << fabmapTrainDataPath << ": FabMap Training Data not found"
                  << std::endl;
        return NULL;
    }
    fs.release();

    //load a chow-liu tree
    std::cout << "Loading Chow-Liu Tree" << std::endl;
    fs.open(chowliutreePath, cv::FileStorage::READ);
    cv::Mat clTree;
    fs["ChowLiuTree"] >> clTree;
    if (clTree.empty()) {
        std::cerr << chowliutreePath << ": Chow-Liu tree not found" <<
                     std::endl;
        return NULL;
    }
    fs.release();

    //create options flags
    std::string newPlaceMethod =
            settings["openFabMapOptions"]["NewPlaceMethod"];
    std::string bayesMethod = settings["openFabMapOptions"]["BayesMethod"];
    int simpleMotionModel = settings["openFabMapOptions"]["SimpleMotion"];
    int options = 0;
    if(newPlaceMethod == "Sampled") {
        options |= of2::FabMap::SAMPLED;
    } else {
        options |= of2::FabMap::MEAN_FIELD;
    }
    if(bayesMethod == "ChowLiu") {
        options |= of2::FabMap::CHOW_LIU;
    } else {
        options |= of2::FabMap::NAIVE_BAYES;
    }
    if(simpleMotionModel) {
        options |= of2::FabMap::MOTION_MODEL;
    }

    of2::FabMap *fabmap;

    //create an instance of the desired type of FabMap
    std::string fabMapVersion = settings["openFabMapOptions"]["FabMapVersion"];
    if(fabMapVersion == "FABMAP1") {
        fabmap = new of2::FabMap1(clTree,
                                  settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options,
                settings["openFabMapOptions"]["NumSamples"]);
    } else if(fabMapVersion == "FABMAPLUT") {
        fabmap = new of2::FabMapLUT(clTree,
                                    settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options,
                settings["openFabMapOptions"]["NumSamples"],
                settings["openFabMapOptions"]["FabMapLUT"]["Precision"]);
    } else if(fabMapVersion == "FABMAPFBO") {
        fabmap = new of2::FabMapFBO(clTree,
                                    settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options,
                settings["openFabMapOptions"]["NumSamples"],
                settings["openFabMapOptions"]["FabMapFBO"]["RejectionThreshold"],
                settings["openFabMapOptions"]["FabMapFBO"]["PsGd"],
                settings["openFabMapOptions"]["FabMapFBO"]["BisectionStart"],
                settings["openFabMapOptions"]["FabMapFBO"]["BisectionIts"]);
    } else if(fabMapVersion == "FABMAP2") {
        fabmap = new of2::FabMap2(clTree,
                                  settings["openFabMapOptions"]["PzGe"],
                settings["openFabMapOptions"]["PzGne"],
                options);
    } else {
        std::cerr << "Could not identify openFABMAPVersion from settings"
                     " file" << std::endl;
        return NULL;
    }

    //add the training data for use with the sampling method
    fabmap->addTraining(fabmapTrainData);

    return fabmap;

}



/*
draws keypoints to scale with coloring proportional to feature strength
*/
void drawRichKeypoints(const cv::Mat& src, std::vector<cv::KeyPoint>& kpts, cv::Mat& dst) {

    cv::Mat grayFrame;
    cvtColor(src, grayFrame, CV_RGB2GRAY);
    cvtColor(grayFrame, dst, CV_GRAY2RGB);

    if (kpts.size() == 0) {
        return;
    }

    std::vector<cv::KeyPoint> kpts_cpy, kpts_sorted;

    kpts_cpy.insert(kpts_cpy.end(), kpts.begin(), kpts.end());

    double maxResponse = kpts_cpy.at(0).response;
    double minResponse = kpts_cpy.at(0).response;

    while (kpts_cpy.size() > 0) {

        double maxR = 0.0;
        unsigned int idx = 0;

        for (unsigned int iii = 0; iii < kpts_cpy.size(); iii++) {

            if (kpts_cpy.at(iii).response > maxR) {
                maxR = kpts_cpy.at(iii).response;
                idx = iii;
            }

            if (kpts_cpy.at(iii).response > maxResponse) {
                maxResponse = kpts_cpy.at(iii).response;
            }

            if (kpts_cpy.at(iii).response < minResponse) {
                minResponse = kpts_cpy.at(iii).response;
            }
        }

        kpts_sorted.push_back(kpts_cpy.at(idx));
        kpts_cpy.erase(kpts_cpy.begin() + idx);

    }

    int thickness = 1;
    cv::Point center;
    cv::Scalar colour;
    int red = 0, blue = 0, green = 0;
    int radius;
    double normalizedScore;

    if (minResponse == maxResponse) {
        colour = CV_RGB(255, 0, 0);
    }

    for (int iii = (int)kpts_sorted.size()-1; iii >= 0; iii--) {

        if (minResponse != maxResponse) {
            normalizedScore = pow((kpts_sorted.at(iii).response - minResponse) / (maxResponse - minResponse), 0.25);
            red = int(255.0 * normalizedScore);
            green = int(255.0 - 255.0 * normalizedScore);
            colour = CV_RGB(red, green, blue);
        }

        center = kpts_sorted.at(iii).pt;
        center.x *= 16;
        center.y *= 16;

        radius = (int)(16.0 * ((double)(kpts_sorted.at(iii).size)/2.0));

        if (radius > 0) {
            circle(dst, center, radius, colour, thickness, CV_AA, 4);
        }

    }

}

/*
Removes surplus features and those with invalid size
*/
void filterKeypoints(std::vector<cv::KeyPoint>& kpts, int maxSize, int maxFeatures) {

    if (maxSize == 0) {
        return;
    }

    sortKeypoints(kpts);

    for (unsigned int iii = 0; iii < kpts.size(); iii++) {

        if (kpts.at(iii).size > float(maxSize)) {
            kpts.erase(kpts.begin() + iii);
            iii--;
        }
    }

    if ((maxFeatures != 0) && ((int)kpts.size() > maxFeatures)) {
        kpts.erase(kpts.begin()+maxFeatures, kpts.end());
    }

}

/*
Sorts keypoints in descending order of response (strength)
*/
void sortKeypoints(std::vector<cv::KeyPoint>& keypoints) {

    if (keypoints.size() <= 1) {
        return;
    }

    std::vector<cv::KeyPoint> sortedKeypoints;

    // Add the first one
    sortedKeypoints.push_back(keypoints.at(0));

    for (unsigned int i = 1; i < keypoints.size(); i++) {

        unsigned int j = 0;
        bool hasBeenAdded = false;

        while ((j < sortedKeypoints.size()) && (!hasBeenAdded)) {

            if (abs(keypoints.at(i).response) > abs(sortedKeypoints.at(j).response)) {
                sortedKeypoints.insert(sortedKeypoints.begin() + j, keypoints.at(i));

                hasBeenAdded = true;
            }

            j++;
        }

        if (!hasBeenAdded) {
            sortedKeypoints.push_back(keypoints.at(i));
        }

    }

    keypoints.swap(sortedKeypoints);

}

void Dijkstra(Graph & graph, int & source)
{
    bool* visited = new bool[graph.vertexNum];
    path[source] = source;
    dist[source] = 0;

    for (int i = 0; i < graph.vertexNum; i++)//初始化dist,path,visited数组
    {
        visited[i] = false;
        if (graph.matrix[source][i]>0 && i != source)//若源顶点source与i直接邻接
        {
            dist[i] = graph.matrix[source][i];
            path[i] = source;
        }
        else//若不是直接邻接，dist置为无穷大
        {
            dist[i] = INT32_MAX;
            path[i] = -1;
        }
    }

    visited[source] = true;

    for (int i = 0; i < graph.vertexNum - 1; i++)//找出除source外剩下的点的最短路径
    {
        int min = INT32_MAX;
        int minPos;
        for (int j = 0; j < graph.vertexNum; j++)//找到权值最小的点
        {
            if (!visited[j] && dist[j] < min)
            {
                min = dist[j];
                minPos = j;
            }
        }

        visited[minPos] = true;
        for (int k = 0; k < graph.vertexNum; k++)//更新dist数组，路径的值
        {
            if (!visited[k] && graph.matrix[minPos][k]>0 && graph.matrix[minPos][k] + min < dist[k])
            {
                dist[k] = graph.matrix[minPos][k] + min;
                path[k] = minPos;
            }
        }
    }

    delete[]visited;
}

void ShowPath(Graph & graph, int & source, int v)
{
    stack<int> s;
    cout << "From here to location" <<"["<< v <<"]" << ", the shortest path is: ";

    while (source != v)
    {
        s.push(v);
        v = path[v];
    }


    cout << "["<< source <<"]";
    while (!s.empty())
    {
        cout << "--" << s.top();
        s.pop();
    }

}
