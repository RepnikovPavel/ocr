#include "stb_image_write.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <vector>
int main(int argc, char** argv){
    const char* out = argc>1?argv[1]:"test.png";
    int W=argc>2?atoi(argv[2]):112, H=argc>3?atoi(argv[3]):112;
    std::vector<unsigned char> img(W*H*3, 255);
    auto setpx=[&](int x,int y,unsigned char r,unsigned char g,unsigned char b){
        if(x<0||x>=W||y<0||y>=H)return;
        img[(y*W+x)*3]=r; img[(y*W+x)*3+1]=g; img[(y*W+x)*3+2]=b;
    };
    for(int x=10;x<W-10;++x){setpx(x,20,0,0,0);setpx(x,22,0,0,0);}
    for(int x=10;x<W-30;++x){setpx(x,50,0,0,0);}
    for(int y=80;y<H-10;++y){setpx(10,y,0,0,0);setpx(W-10,y,0,0,0);}
    for(int x=10;x<W-10;++x){setpx(x,80,0,0,0);setpx(x,H-10,0,0,0);}
    stbi_write_png(out,W,H,3,img.data(),W*3);
    printf("wrote %s %dx%d\n",out,W,H);
    return 0;
}
