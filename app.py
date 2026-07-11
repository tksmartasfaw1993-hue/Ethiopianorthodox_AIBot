def train_and_build_db():
    all_documents = []
    
    # 1. የጠየቅካቸው ሁሉም የኦርቶዶክስ ተዋሕዶ ድኅረ ገጾች ዝርዝር
    web_articles = [
        "https://etartmedia.com",             # የኢቲ አርት ሚዲያ ድኅረ ገጽ
        "https://tewahedomediacenter.org",    # የተዋሕዶ ሚዲያ ማዕከል ድኅረ ገጽ
        "https://debelo.org",                 # የደበሎ ዋና ድኅረ ገጽ
        "https://eotcmk.org",               # የማኅበረ ቅዱሳን ዋና ድኅረ ገጽ
        "https://eotcmk.org",           # የሐመር መጽሔት ገጽ
        "https://ethiopianorthodox.org"  # የEOTC ይፋዊ ገጽ
    ]
    
    # ድኅረ ገጾችን አንድ በአንድ በWebBaseLoader ለማንበብ
    for web_url in web_articles:
        try:
            loader = WebBaseLoader(web_url)
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"የድረገጽ ስህተት ({web_url}): {e}")

    # 2. ሦስቱ ትክክለኛ የዩቲዩብ ቪዲዮዎች (የአባ ገብረኪዳን ቪዲዮን ጨምሮ)
    youtube_videos = [
        "https://youtube.com",
        "https://youtube.com",
        "https://youtube.com"  # የአባ ገብረኪዳን ስብከት ቪዲዮ
    ]
    
    # የዩቲዩብ ቪዲዮዎችን ጽሑፍ (Transcript) አንድ በአንድ መጫን
    for video_url in youtube_videos:
        try:
            loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
            all_documents.extend(loader.load())
        except Exception as e:
            print(f"የዩቲዩብ ስህተት ({video_url}): {e}")

    # 3. መረጃው በስህተት ባዶ እንዳይሆንና አፑ እንዳይዘጋ መከላከያ
    if not all_documents:
        st.error("ምንም መረጃ አልተገኘም! እባክዎ የሊንኮቹን ትክክለኛነት ወይም የኢንተርኔት ግንኙነትዎን ያረጋግጡ።")
        return None

    # 4. የተጫኑትን ጽሑፎች ለአይአዩ (AI) ምቹ እንዲሆኑ መበተን (Text Splitting)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(all_documents)
    
    # ማሳሰቢያ፦ ከዚህ በታች የእርስዎ የChroma/FAISS ቬክተር ዳታቤዝ ግንባታ ኮድ ይቀጥላል...
