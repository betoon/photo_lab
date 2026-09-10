// PhotoLab's isolated Nikon SDK bridge. SDK headers and binaries remain external.
#include <windows.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#include <string>
#include "Nkfl_Interface.h"
static std::string utf8(const wchar_t* s) {
    int n=WideCharToMultiByte(CP_UTF8,0,s,-1,NULL,0,NULL,NULL);
    std::string out(n,0); WideCharToMultiByte(CP_UTF8,0,s,-1,out.data(),n,NULL,NULL); return out;
}
int wmain(int argc,wchar_t** argv) {
    if(argc!=5) { fprintf(stderr,"Expected NEF path, output path, bit depth, profile directory\n"); return 2; }
    HMODULE dll=LoadLibraryW(L"NkImgSDK.dll");
    if(!dll) { fprintf(stderr,"Cannot load Nikon SDK (%lu)\n",GetLastError()); return 3; }
    auto entry=(Nkfl_EntryProcPtr)GetProcAddress(dll,"Nkfl_Entry");
    if(!entry) return 3;
    NkflLibraryParam lib={}; lib.ulSize=sizeof(lib);lib.ulVersion=0x01000000;
    auto profiles=utf8(argv[4]); strncpy_s((char*)lib.DefProfPath,MAX_PATH,profiles.c_str(),_TRUNCATE);
    NkflSessionParam session={};session.ulSize=sizeof(session);
    bool opened=false; int result=1;
    auto check=[&](unsigned long command,void* p) {
        auto code=entry(command,p);
        if(code) { fprintf(stderr,"Nikon SDK command 0x%lx failed: 0x%lx\n",command,code); throw code; }
    };
    try {
        check(kNkfl_Cmd_OpenLibrary,&lib);opened=true;
        NkflDevelopColorMode mode={};mode.ulSize=sizeof(mode);mode.lDevelopColorMode=1;
        check(kNkfl_Cmd_SetDevelopColorMode,&mode);
        auto input=utf8(argv[1]);session.ulType=kNkfl_Source_FileName_UTF8;session.pFileInfo=input.data();
        check(kNkfl_Cmd_OpenSession,&session);
        NkflOutputDeviceProfile profile={};profile.ulSize=sizeof(profile);profile.ulSessionID=session.ulSessionID;
        auto srgb=profiles; if(!srgb.empty() && srgb.back()==0) srgb.pop_back(); srgb+="\\NKsRGB.icm";
        strncpy_s((char*)profile.OutputDeviceProfile,MAX_PATH,srgb.c_str(),_TRUNCATE);
        check(kNkfl_Cmd_SetOutputDeviceProfile,&profile);
        NkflImageInfoParam info={};info.ulSize=sizeof(info);info.ulSessionID=session.ulSessionID;
        check(kNkfl_Cmd_GetOriginalInfo,&info);
        info.ulByteDepth=wcscmp(argv[3],L"16")==0?2:1;
        // Set dimensions before rotation, then query the oriented output dimensions.
        check(kNkfl_Cmd_SetImageInfo,&info);check(kNkfl_Cmd_GetImageInfo,&info);
        uint64_t length=(uint64_t)info.ulWidth*info.ulHeight*3*info.ulByteDepth;
        if(!length || length>1024ULL*1024*1024) throw (unsigned long)4;
        std::vector<unsigned char> pixels((size_t)length);
        NkflImageParam data={};data.ulSize=sizeof(data);data.ulSessionID=session.ulSessionID;
        data.rectArea.right=info.ulWidth;data.rectArea.bottom=info.ulHeight;
        data.ulDataSize=(unsigned long)length;data.pData=pixels.data();
        check(kNkfl_Cmd_GetImageData,&data);
        if(data.ulDataSize!=length) throw (unsigned long)4;
        FILE* out=nullptr;
        if(_wfopen_s(&out,argv[2],L"wb") || !out) throw (unsigned long)5;
        uint32_t header[4]={0x4e4b5247,info.ulWidth,info.ulHeight,info.ulByteDepth};
        bool ok=fwrite(header,sizeof(header),1,out)==1 && fwrite(pixels.data(),1,pixels.size(),out)==pixels.size();
        if(fclose(out)!=0) ok=false;
        if(!ok) throw (unsigned long)5;
        result=0;
    } catch(...) { fprintf(stderr,"Nikon RAW development failed\n"); }
    if(session.ulSessionID) entry(kNkfl_Cmd_CloseSession,&session);
    if(opened) entry(kNkfl_Cmd_CloseLibrary,nullptr);
    FreeLibrary(dll);return result;
}
