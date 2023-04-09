int** generatedMatrix(int n,int* returnSize,int ** returnColumnSizes){
    *returnSize=n;
    *returnColumnSizes=(int*)malloc(sizeof(int)*n);
    int** ans=(int**)malloc(sizeof(int*)*n);
    int i;
    for(i=0;i<n;i++){
        ans[i]=(int*)malloc(sizeof(int)*n);
        (*returnColumnSizes)[i]=n;
    }
    int startX=0;
    int startY=0;
    int mid=n/2;
    int loop=n/2;
    int offset=1;
    while(loop){
        int i=startX;
        int j=startY;
        for(;j<startY+n-offset;j++){
            ans[startS][j]=count++;
        }
    }

    //unremembered
    

}