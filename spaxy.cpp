#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <limits>
#include <ctime>
#include <random>
using namespace sycl;
using namespace std;
void initVector(vector<float>& v, bool ifRandom)
{
    if(ifRandom)
    {
        uniform_real_distribution<float> u(-100, 100);
        default_random_engine e(time(NULL));
        for(int i=0;i<v.size();i++)
            v[i]=u(e);
    }
    else
    {
        for(int i=0;i<v.size();i++)
            v[i]=0;
    }
}
bool checkResult(vector<float>& h,vector<float>& d)
{
    bool flag=true;
    for(int i=0;i<h.size();i++)
    {
        if(fabs(h[i]-d[i])>1e-5)
        {
            cout<<"h_r["<<i<<"]:"<<h[i]<<" vs d_r["<<i<<"]:"<<d[i]<<"\n";
            flag=false;
        }
    }
    return flag?true:false;
}
int main(int argc,char * argv[])
{
    int n = 1<<20;
    float a = 2.0f;
    
    clock_t start, finish; 
    double  duration;
    vector<float> h_x(n), h_y(n),h_r(n),d_r(n);
    buffer b_x{h_x}, b_y{h_y},b_r{d_r};
    initVector(h_x,true);
    initVector(h_y,true);
    initVector(h_r,false);
    initVector(d_r,false);
    auto selector = default_selector_v;
    queue q(selector);
    device my_device=q.get_device();
    cout<<"Device:"<<my_device.get_info<info::device::name>()<<"\n";
    cout<<"Problem Size: "<<n<<"\n";
   
    q.submit([&](handler &h){
        auto x=b_x.get_access(h,read_only);
        auto y=b_y.get_access(h,read_only);
        auto r=b_r.get_access(h,write_only);
        start = clock();
        h.parallel_for(n,[=](auto i){
            r[i] = a*x[i]+y[i];
        });
       
    }).wait();
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<"device finished in "<<duration<<"seconds\n";
    start = clock();
    for(int i=0;i<n;i++)
    {
        h_r[i] = a*h_x[i]+h_y[i];
    }
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    cout<<"host finished in "<<duration<<"seconds\n";

    bool flag = checkResult(h_r,d_r);
    if(flag)
    {
        cout<<"check result passed!\n";
    }
    else
    {
        cout<<"check result failed!\n";
    }
    // for(int i=0;i<n;i++)
    //     cout<<"r["<<i<<"]="<<d_r[i]<<"\n";
    return 0;
}