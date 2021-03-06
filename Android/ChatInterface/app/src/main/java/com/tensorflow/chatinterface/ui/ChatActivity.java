package com.tensorflow.chatinterface.ui;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.widget.Toolbar;
import android.text.TextUtils;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.view.inputmethod.InputMethodManager;
import android.widget.ArrayAdapter;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ListView;
import android.widget.TabHost;
import android.widget.TextView;
import android.widget.Toast;

import com.iflytek.cloud.ErrorCode;
import com.iflytek.cloud.InitListener;
import com.iflytek.cloud.RecognizerListener;
import com.iflytek.cloud.RecognizerResult;
import com.iflytek.cloud.SpeechConstant;
import com.iflytek.cloud.SpeechError;
import com.iflytek.cloud.SpeechRecognizer;
import com.iflytek.cloud.SpeechSynthesizer;
import com.iflytek.cloud.SynthesizerListener;
import com.iflytek.cloud.ui.RecognizerDialog;
import com.iflytek.cloud.ui.RecognizerDialogListener;
import com.iflytek.sunflower.FlowerCollector;
import com.tensorflow.chatinterface.ChatApplication;
import com.tensorflow.chatinterface.R;
import com.tensorflow.chatinterface.util.HttpUtils;
import com.tensorflow.chatinterface.util.JsonParser;
import com.tensorflow.chatinterface.util.StringUtils;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;

public class ChatActivity extends BaseActivity {
    private static final String TAG = ChatActivity.class.getSimpleName();

    private static final int REQUEST_PERMISSION = 1;
    private final int VIEW_TYPE = 0xb01;
    private final int MESSAGE = 0xb02;
    private final int VIEW_TYPE_LEFT = -10;
    private final int VIEW_TYPE_RIGHT = -11;
    private ListView mListView;
    private MessageAdapter mAdapter;
    private EditText mEtMessageInput;
    private ImageView mBtnSend, mBtnPhonetics;
    private ArrayList<HashMap<Integer, Object>> mItems = null;
    private Toolbar mToolbar;

    //??????????????????
    private SpeechRecognizer mSpeechRecognizer;
    //???????????????
    private String voicer = "xiaoyan";
    //????????????UI
    private RecognizerDialog mRecognizerDialog;
    //???hashmap????????????????????????
    private HashMap<String, String> mIatResults = new LinkedHashMap<>();
    //??????????????????
    private SpeechSynthesizer mSpeechSynthesizer;
    //????????????
    private int mPercentForBuffering = 0;
    //????????????
    private int mPercentForPlaying = 0;

    //??????????????????
    private SharedPreferences mSharedPreferences;

    private String mEngineType = SpeechConstant.TYPE_CLOUD;
    private boolean mTranslateEnale = false;
    private int mRet = 0;
    private InputMethodManager mImManager;

    private long mLastTime;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        getWindow().setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE);
        setContentView(R.layout.base_chart);
        super.onCreate(savedInstanceState);

        checkPermissions();
        initToolbar();
        initViews();
        initParams();
    }

    private void initParams() {
        //???????????????????????????
        mSpeechRecognizer = SpeechRecognizer.createRecognizer(this, mInitListener);
        //????????????????????????Dialog
        mRecognizerDialog = new RecognizerDialog(this, mInitListener);
        mSharedPreferences = getSharedPreferences(IatSettings.PREFER_NAME, Activity.MODE_PRIVATE);
        //?????????????????????
        mSpeechSynthesizer = SpeechSynthesizer.createSynthesizer(this,mTtsInitListener);
        mImManager = (InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
    }

    /**
     * ?????????????????????????????????
     */
    private InitListener mInitListener = new InitListener() {
        @Override
        public void onInit(int code) {
            Log.d(TAG, "SpeechRecognizer init() code=" + code);
            if (code != ErrorCode.SUCCESS){
                Toast.makeText(getApplicationContext(),"?????????????????????????????????" + code,Toast.LENGTH_LONG).show();
            }
        }
    };



    /**
     * ?????????????????????????????????
     */
    private InitListener mTtsInitListener = new InitListener() {
        @Override
        public void onInit(int code) {
            Log.d(TAG, "InitListener init() code=" + code);
            if (code != ErrorCode.SUCCESS){
                Toast.makeText(getApplicationContext(),"?????????????????????????????????" + code,Toast.LENGTH_LONG).show();
            }
        }
    };

    /**
     * ??????UI?????????
     */
    private RecognizerDialogListener mRecognizerDialogListener = new RecognizerDialogListener() {
        @Override
        public void onResult(RecognizerResult recognizerResult, boolean b) {
            if (mTranslateEnale){
                printTranslateResult(recognizerResult);
                Log.d(TAG, "??????UI??????????????????");
            }else {
                printResult(recognizerResult);
                Log.d(TAG, "??????UI??????????????????");
            }
        }

        @Override
        public void onError(SpeechError speechError) {
            if(mTranslateEnale && speechError.getErrorCode() == 14002){
                Toast.makeText(getApplicationContext(),speechError.getPlainDescription(true)+"\n??????????????????????????????????????????", Toast.LENGTH_LONG).show();
            }else {
                Toast.makeText(getApplicationContext(), speechError.getPlainDescription(true),Toast.LENGTH_LONG).show();
            }
        }
    };

    private SynthesizerListener mTtsListener = new SynthesizerListener() {
        @Override
        public void onSpeakBegin() {
            Toast.makeText(getApplicationContext(),"????????????", Toast.LENGTH_LONG).show();
        }

        @Override
        public void onBufferProgress(int i, int i1, int i2, String s) {

        }

        @Override
        public void onSpeakPaused() {
            Toast.makeText(getApplicationContext(),"????????????", Toast.LENGTH_LONG).show();
        }

        @Override
        public void onSpeakResumed() {
            Toast.makeText(getApplicationContext(),"????????????", Toast.LENGTH_LONG).show();

        }

        @Override
        public void onSpeakProgress(int i, int i1, int i2) {

        }

        @Override
        public void onCompleted(SpeechError speechError) {
            if (speechError == null){
                Toast.makeText(getApplicationContext(),"????????????",Toast.LENGTH_LONG).show();
            }else {
                Toast.makeText(getApplicationContext(),speechError.getPlainDescription(true),Toast.LENGTH_LONG).show();
            }
        }

        @Override
        public void onEvent(int i, int i1, int i2, Bundle bundle) {

        }
    };

    private void printResult(RecognizerResult results) {
        String text = JsonParser.parseIatResult(results.getResultString());
        String sn = null;
        try {
            JSONObject resultJSON = new JSONObject(results.getResultString());
            sn = resultJSON.optString("sn");
        }catch (JSONException e){
            e.printStackTrace();
        }
        mIatResults.put(sn, text);
        StringBuffer resultBuffer = new StringBuffer();
        for (String key : mIatResults.keySet()){
            resultBuffer.append(mIatResults.get(key));
        }
        String message = resultBuffer.toString();
        long currentTime = System.currentTimeMillis();
        if (currentTime - mLastTime > 1500) {
            refreshUi(message, VIEW_TYPE_RIGHT);
            mLastTime = currentTime;
        }
    }

    private void printTranslateResult(RecognizerResult results) {
        String trans = JsonParser.parseTransResult(results.getResultString(), "dst");
        String oris = JsonParser.parseTransResult(results.getResultString(),"src");

        if (TextUtils.isEmpty(trans) || TextUtils.isEmpty(oris)){
            Toast.makeText(getApplicationContext(),"????????????????????????????????????????????????????????????",Toast.LENGTH_LONG).show();
        }
    }

    /**
     * ???????????????
     */
    private RecognizerListener mRecognizerListener = new RecognizerListener() {
        @Override
        public void onVolumeChanged(int i, byte[] bytes) {
            Toast.makeText(getApplicationContext(),"???????????????????????????????????????"+i,Toast.LENGTH_LONG).show();
            Log.d(TAG,"?????????????????????"+ bytes.length);
        }

        @Override
        public void onBeginOfSpeech() {
            Toast.makeText(getApplicationContext(),"????????????",Toast.LENGTH_LONG).show();
        }

        @Override
        public void onEndOfSpeech() {
            Toast.makeText(getApplicationContext(),"????????????",Toast.LENGTH_LONG).show();
        }

        @Override
        public void onResult(RecognizerResult recognizerResult, boolean isLast) {
            if(mTranslateEnale){
                printTranslateResult(recognizerResult);
                Log.d(TAG,"??????????????????????????????");
            }else {
                printResult(recognizerResult);
                Log.d(TAG,"??????????????????????????????");
            }

            if(isLast){
                //TODO ?????????????????????
            }
        }

        @Override
        public void onError(SpeechError speechError) {
            if (mTranslateEnale && speechError.getErrorCode() == 14002){
                Toast.makeText(getApplicationContext(),speechError.getPlainDescription(true) +"\n??????????????????????????????????????????",Toast.LENGTH_LONG).show();

            }else {
                Toast.makeText(getApplicationContext(),speechError.getPlainDescription(true),Toast.LENGTH_LONG).show();
            }
        }

        @Override
        public void onEvent(int i, int i1, int i2, Bundle bundle) {

        }
    };

    /**
     * ????????????????????????
     */
    public void setSpeechParam(){
        //????????????
        mSpeechRecognizer.setParameter(SpeechConstant.PARAMS, null);
        //?????????????????????
        mSpeechRecognizer.setParameter(SpeechConstant.ENGINE_TYPE, mEngineType);
        //???????????????????????????
        mSpeechRecognizer.setParameter(SpeechConstant.RESULT_TYPE, "json");

        this.mTranslateEnale = mSharedPreferences.getBoolean(this.getString(R.string.pref_key_translate), false);
        if (mTranslateEnale){
            mSpeechRecognizer.setParameter(SpeechConstant.ASR_SCH, "1");
            mSpeechRecognizer.setParameter(SpeechConstant.ADD_CAP, "translate");
            mSpeechRecognizer.setParameter(SpeechConstant.TRS_SRC, "its");
        }

        //????????????
        String lag = mSharedPreferences.getString("iat_language_preference","mandarin");
        if (lag.equals("en_us")){
            mSpeechRecognizer.setParameter(SpeechConstant.LANGUAGE,"en_us");
            mSpeechRecognizer.setParameter(SpeechConstant.ACCENT, null);

            if (mTranslateEnale){
                mSpeechRecognizer.setParameter(SpeechConstant.ORI_LANG,"en");
                mSpeechRecognizer.setParameter(SpeechConstant.ACCENT,"cn");
            }
        }else {
            mSpeechRecognizer.setParameter(SpeechConstant.LANGUAGE,"zh_cn");
            mSpeechRecognizer.setParameter(SpeechConstant.ACCENT, lag);
            if (mTranslateEnale){
                mSpeechRecognizer.setParameter(SpeechConstant.ORI_LANG,"cn");
                mSpeechRecognizer.setParameter(SpeechConstant.ACCENT,"en");
            }
        }
        //??????????????????????????????????????????????????????????????????????????????????????????
        mSpeechRecognizer.setParameter(SpeechConstant.VAD_BOS, mSharedPreferences.getString("iat_vadbos_preference","4000"));
        //???????????????????????????????????????????????????????????????????????????????????????????????????????????????
        mSpeechRecognizer.setParameter(SpeechConstant.VAD_EOS, mSharedPreferences.getString("iat_vadeos_preference","1000"));
        mSpeechRecognizer.setParameter(SpeechConstant.ASR_PTT, mSharedPreferences.getString("iat_punc_preference","1"));

        //??????????????????
        mSpeechRecognizer.setParameter(SpeechConstant.AUDIO_FORMAT,"wav");
        mSpeechRecognizer.setParameter(SpeechConstant.ASR_AUDIO_PATH, Environment.getExternalStorageDirectory() + "/msc/iat.wav");
    }

    /**
     * ????????????????????????
     */
    public void setVoiceParam(){
        mSpeechSynthesizer.setParameter(SpeechConstant.PARAMS, null);
        if(mEngineType.equals(SpeechConstant.TYPE_CLOUD)){
            mSpeechSynthesizer.setParameter(SpeechConstant.ENGINE_TYPE, SpeechConstant.TYPE_CLOUD);
            mSpeechSynthesizer.setParameter(SpeechConstant.VOICE_NAME, voicer);
            mSpeechSynthesizer.setParameter(SpeechConstant.SPEED, mSharedPreferences.getString("speed_preference","50"));
            mSpeechSynthesizer.setParameter(SpeechConstant.PITCH,mSharedPreferences.getString("pitch_preference","50"));
            mSpeechSynthesizer.setParameter(SpeechConstant.VOLUME,mSharedPreferences.getString("volume_preference","50"));
        }else {
            mSpeechSynthesizer.setParameter(SpeechConstant.ENGINE_TYPE, SpeechConstant.TYPE_LOCAL);
            mSpeechSynthesizer.setParameter(SpeechConstant.VOICE_NAME,"");
        }
        mSpeechSynthesizer.setParameter(SpeechConstant.STREAM_TYPE, mSharedPreferences.getString("stram_preference","3"));
        mSpeechSynthesizer.setParameter(SpeechConstant.KEY_REQUEST_FOCUS,"true");

        //??????????????????
        mSpeechRecognizer.setParameter(SpeechConstant.AUDIO_FORMAT,"wav");
        mSpeechRecognizer.setParameter(SpeechConstant.ASR_AUDIO_PATH, Environment.getExternalStorageDirectory() + "/msc/tts.wav");

    }


    /**
     * ??????Android 6.0????????????????????????
     */
    private void checkPermissions() {
        List<String> permissionList = new ArrayList<>();
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.ACCESS_COARSE_LOCATION);
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.ACCESS_FINE_LOCATION);
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.RECORD_AUDIO);
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.READ_PHONE_STATE);
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.READ_CONTACTS);
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            permissionList.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
        }
        if (!permissionList.isEmpty()) {
            String[] permissions = permissionList.toArray(new String[permissionList.size()]);
            ActivityCompat.requestPermissions(this, permissions, REQUEST_PERMISSION);
        } else {
            Log.d(TAG, "??????????????????");
        }
    }

    /**
     * ?????????Toolbar
     */
    private void initToolbar() {
        mToolbar = findViewById(R.id.toolbar);
        setSupportActionBar(mToolbar);
        ((TextView) mToolbar.findViewById(R.id.title_toolbar)).setText("ChatRobot");
    }

    /**
     * ?????????????????????
     */
    private void initViews() {
        mItems = new ArrayList<>();
        mListView = findViewById(android.R.id.list);
        mAdapter = new MessageAdapter(this, -1);
        mListView.setAdapter(mAdapter);
        mEtMessageInput = findViewById(R.id.edit_send);
        mBtnSend = findViewById(R.id.btn_send);
        mBtnSend.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                String message = mEtMessageInput.getText().toString();
                if(TextUtils.isEmpty(message)){
                    Toast.makeText(ChatActivity.this, "????????????????????????", Toast.LENGTH_SHORT).show();
                    return;
                }
                mEtMessageInput.setText(null);
                refreshUi(message, VIEW_TYPE_RIGHT);
            }
        });
        mBtnPhonetics = findViewById(R.id.btn_phonetics);
        mBtnPhonetics.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                FlowerCollector.onEvent(ChatActivity.this, "iat_recognize");
                mIatResults.clear();
                setSpeechParam();
                boolean isShowDialog = mSharedPreferences.getBoolean(getString(R.string.pref_key_iat_show), true);
                if (isShowDialog){
                    mRecognizerDialog.setListener(mRecognizerDialogListener);
                    mRecognizerDialog.show();
                    Toast.makeText(getApplicationContext(),R.string.text_begin,Toast.LENGTH_LONG).show();
                }else {
                    mRet = mSpeechRecognizer.startListening(mRecognizerListener);
                    if(mRet != ErrorCode.SUCCESS){
                        Toast.makeText(getApplicationContext(),"??????????????????????????????"+mRet,Toast.LENGTH_LONG).show();
                    }else {
                        Toast.makeText(getApplicationContext(), R.string.text_begin,Toast.LENGTH_LONG).show();
                    }

                }
            }
        });
    }

    /**
     * ????????????????????????
     *
     * @param message
     */
    private void msgToTfLite(final String message) {
        final String received = "";
//        if (message.equals(StringUtils.sSendTo[0])) {
//            received = StringUtils.sSendReceived[0];
//        } else if (message.equals(StringUtils.sSendTo[1])) {
//            received = StringUtils.sSendReceived[1];
//        } else if (message.equals(StringUtils.sSendTo[2])) {
//            received = StringUtils.sSendReceived[2];
//        } else if (message.equals(StringUtils.sSendTo[3])) {
//            received = StringUtils.sSendReceived[3];
//        } else if (message.equals(StringUtils.sSendTo[4])) {
//            received = StringUtils.sSendReceived[4];
//        } else if (message.equals(StringUtils.sSendTo[5])) {
//            received = StringUtils.sSendReceived[5];
//        } else {
//            received = message;
//        }
        new Thread(new Runnable() {
            @Override
            public void run() {
                final String received = HttpUtils.getInstance().getRequest(
                        HttpUtils.getInstance().getRequestUrl(ChatApplication.URL, message));
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        refreshUi(received.split("<")[0]);
                    }
                });
            }
        }).start();


    }

    private void refreshUi(String received){
        refreshUi(received, VIEW_TYPE_LEFT);
        setVoiceParam();
        int code = mSpeechSynthesizer.startSpeaking(received, mTtsListener);
        if (code != ErrorCode.SUCCESS){
            Toast.makeText(getApplicationContext(),"????????????????????????????????????"+code,Toast.LENGTH_LONG).show();
        }
    }

    /**
     * ????????????????????????
     *
     * @param msg
     */
    private void refreshUi(String msg, int msgType) {
        if (TextUtils.isEmpty(msg)) {
            Toast.makeText(this, "????????????|-???-|try again later", Toast.LENGTH_SHORT).show();
            return;
        }
        HashMap<Integer, Object> map = new HashMap<>();
        map.put(VIEW_TYPE, msgType);
        map.put(MESSAGE, msg);
        mItems.add(map);
        mAdapter.notifyDataSetChanged();
        if (msgType == VIEW_TYPE_RIGHT) {
            msgToTfLite(msg);
        }
    }

    /**
     * ListView???Adapter
     */
    private class MessageAdapter extends ArrayAdapter {
        private LayoutInflater layoutInflater;

        public MessageAdapter(Context context, int resource) {
            super(context, resource);
            layoutInflater = LayoutInflater.from(context);
        }

        @Override
        public View getView(int pos, View convertView, ViewGroup parent) {
            int type = getItemViewType(pos);
            String msg = getItem(pos);
            switch (type) {
                case VIEW_TYPE_LEFT:
                    convertView = layoutInflater.inflate(R.layout.base_left_usr, null);
                    TextView textLeft = convertView.findViewById(R.id.usr_msg);
                    textLeft.setText(msg);
                    break;

                case VIEW_TYPE_RIGHT:
                    convertView = layoutInflater.inflate(R.layout.base_right_usr, null);
                    TextView textRight = convertView.findViewById(R.id.usr_msg);
                    textRight.setText(msg);
                    break;
            }
            return convertView;
        }

        @Override
        public String getItem(int pos) {
            String s = mItems.get(pos).get(MESSAGE) + "";
            return s;
        }

        @Override
        public int getCount() {
            return mItems.size();
        }

        @Override
        public int getItemViewType(int pos) {
            int type = (Integer) mItems.get(pos).get(VIEW_TYPE);
            return type;
        }

        @Override
        public int getViewTypeCount() {
            return 2;
        }

    }
}
