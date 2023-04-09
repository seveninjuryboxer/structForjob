int search(int* nums,int numsSize,int target){
    int left=0;
    int length=numsSize;
    int right=length;
    int middle=0;
    while(left<right){
        //there is a fault
        int middle=left+(right-left)/2;
        if(nums[middle]<target){
            left=middle+1;
        }
        else if(nums[middle]>target){
            right=middle;
        }
        else{
            return middle;
        }
    }
    return -1;
}