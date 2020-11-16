#include <bits/stdc++.h>
#include <iostream>

using namespace std;
const int V=1100;
int n,m,k,x,y,pre[V];
bool v[V],a[V][V];

bool dfs(int i)
{
    for(int j=1;j<=m;j++)
    if((!v[j])&&(a[i][j])){
        v[j]=1;
        if(pre[j]==0 || dfs(pre[j])){
            pre[j]=i;
            return 1;
        }
    }
    return 0;
}

int main()
{
    cin>>n>>m>>k;
    memset(a,0,sizeof(a));
    memset(pre,0,sizeof(pre));
    for(int i=1;i<=k;i++){
        cin>>x>>y;
        a[x][y]=1;
    }
    int ans=0;
    for(int i=1;i<=n;i++){
        memset(v,0,sizeof(v));
        if(dfs(i))
        ans++;
    }
    cout<<"Result is "<<n+m-ans<<"."<<endl;
    return 0;
}
