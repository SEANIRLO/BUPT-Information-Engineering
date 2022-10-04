/*
 * Copyright (c) 2021 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified by Sean XIAO, May 2022
 * For academic purposes ONLY
 */

package com.example.bburecog.slice;

import com.example.bburecog.utils.Base64Util;
import com.example.bburecog.utils.ThreadPoolUtil;
import com.example.bburecog.DetectAbility;
import com.example.bburecog.ResourceTable;

import ohos.agp.components.element.ShapeElement;
import ohos.agp.render.Paint;
import ohos.agp.utils.Color;
import ohos.eventhandler.EventHandler;
import ohos.eventhandler.EventRunner;
import ohos.eventhandler.InnerEvent;
import ohos.net.*;
import ohos.aafwk.content.Operation;
import ohos.agp.components.*;
import ohos.agp.window.dialog.CommonDialog;
import ohos.aafwk.ability.AbilitySlice;
import ohos.aafwk.content.Intent;
import ohos.aafwk.ability.DataAbilityHelper;
import ohos.hiviewdfx.HiLog;
import ohos.hiviewdfx.HiLogLabel;
import ohos.media.image.ImageSource;
import ohos.media.image.PixelMap;
import ohos.media.photokit.metadata.AVStorage;
import ohos.utils.net.Uri;
import ohos.agp.components.Component;
import ohos.agp.components.Text;

import java.io.*;
import java.net.*;

import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;

import static com.example.bburecog.DetectAbility.ACTION;

/**
 * DetectAbilityMainSlice
 */
public class DetectAbilityMainSlice extends AbilitySlice {
    // HiLog Settings - Printing from emulator is not available
    static final HiLogLabel label = new HiLogLabel(HiLog.LOG_APP, 0x0001, "DETECT TEST");
    private static final String TAG = DetectAbilityMainSlice.class.getSimpleName();
    private static final HiLogLabel LABEL_LOG = new HiLogLabel(3, 0xD000F00, TAG);

    // Event ID
    private final int REQUEST_CODE = 1101;
    private static final int DETECT_SUCCESS = 1200;
    private static final int DETECT_NO_RESULT = 1201;

    // Component
    private Button chooseImgButton;
    private Button detectButton;
    private Image chosenImage;
    private String imgStr;
    private Text resultText;
    private Text logoutText;
    private Text historyText;
    private CommonDialog statusDialog;

    // Internet Manager
    private NetManager netManager;

    // Result
    private String res;
    private int prob;
    private JSONObject result;

    // Current number of the animation for the circle progress
    private int roateNum = 0;

    // Instance a EventHandler get the result of detection, Internet connection error omitted
    private final EventHandler detectEventHandler =
            new EventHandler(EventRunner.getMainEventRunner()) {
                @Override
                protected void processEvent(InnerEvent event) {
                    super.processEvent(event);
                    showProgress(false);
                    switch (event.eventId) {
                        case DETECT_SUCCESS:
                            JSONArray arr = result.getJSONArray("result");
                            float max_prob = 0;
                            int index = -1;
                            for(int i=0;i<arr.size();i++){ // Choose ONE result with maximum probability
                                if(arr.getJSONObject(i).getFloat("probability") >= max_prob){
                                    index = i;
                                    max_prob = arr.getJSONObject(i).getFloat("probability");
                                }
                            }
                            res = arr.getJSONObject(index).getString("name");
                            prob = Math.round(arr.getJSONObject(index).getFloat("probability") * 100);
                            getUITaskDispatcher().asyncDispatch(() -> resultText.setText(res + ", " + prob + "%"));
                            break;
                        case DETECT_NO_RESULT:
                            showDetectDialog(false);
                            break;
                        default:
                            break;
                    }
                }
            };

    /**
     * The initComponents, get component from xml
     */
    private void initComponents() {
        chooseImgButton = (Button) findComponentById(ResourceTable.Id_choose_img_button);
        detectButton = (Button)findComponentById(ResourceTable.Id_detect_button);
        chosenImage = (Image)findComponentById(ResourceTable.Id_show_chosen_image);
        chosenImage.setPixelMap(ResourceTable.Media_imageArea);
        resultText = (Text)findComponentById(ResourceTable.Id_resultText);
        logoutText = (Text) findComponentById(ResourceTable.Id_logoutText);
        historyText = (Text) findComponentById(ResourceTable.Id_historyText);
    }

    /**
     * The initListener, set listener of component
     */
    private void initListener(){
        chooseImgButton.setClickedListener(component -> selectPic());
        detectButton.setClickedListener(this::detect);
        logoutText.setClickedListener(component -> showLogoutDialog());
        historyText.setClickedListener(this::startAbilitySlice);
    }

    /**
     * Get image content from album or storage
     */
    private void selectPic(){
        Intent intent = new Intent();
        Operation opt = new Intent.OperationBuilder().withAction("android.intent.action.GET_CONTENT").build();
        intent.setOperation(opt);
        intent.addFlags(Intent.FLAG_NOT_OHOS_COMPONENT);
        intent.setType("image/*");
        startAbilityForResult(intent, REQUEST_CODE);
    }

    /**
     * Detect Online
     */
    private void detect(Component component){
        String accessToken = "24.02170c2ee1703ddf6e7ce4a0619f2930.2592000.1655313810.282335-26141380"; // Valid for 30 days
        String requestUrl = "https://aip.baidubce.com/rest/2.0/image-classify/v2/logo";
        String urlString = requestUrl + "?access_token=" + accessToken;
        String contentType = "application/x-www-form-urlencoded";
        String encoding = "UTF-8";

        showProgress(true);
        netManager = NetManager.getInstance(null);
        if (!netManager.hasDefaultNet()) {
            return;
        }
        ThreadPoolUtil.submit(() -> {
            NetHandle netHandle = netManager.getDefaultNet();
            netManager.addDefaultNetStatusCallback(callback);
            HttpURLConnection connection = null;

            try {
                String imgParam = URLEncoder.encode(imgStr, "UTF-8");
                String params = "image=" + imgParam + "&custom_lib=" + false;
                URL url = new URL(urlString);
                // Open the connection with URL
                URLConnection urlConnection = netHandle.openConnection(url, java.net.Proxy.NO_PROXY);
                if (urlConnection instanceof HttpURLConnection) {
                    connection = (HttpURLConnection) urlConnection;
                }
                connection.setConnectTimeout(500);
                connection.setRequestMethod("POST");
                // Set the attributes of request
                connection.setRequestProperty("Content-Type", contentType);
                connection.setRequestProperty("Connection", "Keep-Alive");
                connection.setUseCaches(false);
                connection.setDoOutput(true);
                connection.setDoInput(true);
                // Get output stream
                DataOutputStream out = new DataOutputStream(connection.getOutputStream());
                out.write(params.getBytes(encoding));
                out.flush();
                out.close();
                // Setup connection
                // For further development - detectEventHandler.sendEvent(DETECT_CONNECTION_FAILED);
                connection.connect();

                // Read the response of URL
                BufferedReader in = null;
                in = new BufferedReader(
                        new InputStreamReader(connection.getInputStream(), encoding));
                String rawResult = "";
                String getLine;
                while ((getLine = in.readLine()) != null) {
                    rawResult += getLine;
                }
                in.close();

                // Parse the returned json result
                result = JSONObject.parseObject(rawResult);
                if(result.getInteger("result_num") == 0){
                    detectEventHandler.sendEvent(DETECT_NO_RESULT);
                }
                else{
                    detectEventHandler.sendEvent(DETECT_SUCCESS);
                }
                HttpResponseCache.getInstalled().flush();
            } catch (IOException e) {
                HiLog.error(LABEL_LOG, "%{public}s", "netRequest IOException");
            }
        });
    }

    /**
     * Net Status
     */
    private final NetStatusCallback callback = new NetStatusCallback() {
        @Override
        public void onAvailable(NetHandle handle) {
            HiLog.info(LABEL_LOG, "%{public}s", "NetStatusCallback onAvailable");
        }

        @Override
        public void onBlockedStatusChanged(NetHandle handle, boolean blocked) {
            HiLog.info(LABEL_LOG, "%{public}s", "NetStatusCallback onBlockedStatusChanged");
        }
    };

    /**
     * The showLogoutDialog, confirm that whether the user wants to log out or not
     */
    private void showLogoutDialog() {
        // Init dialog
        CommonDialog logoutDialog = new CommonDialog(this);
        // Get component from xml
        Component layoutComponent =
                LayoutScatter.getInstance(this).parse(ResourceTable.Layout_warning_dialog, null, false);
        Text dialogText = (Text) layoutComponent.findComponentById(ResourceTable.Id_dialog_text);
        Text dialogSubText = (Text) layoutComponent.findComponentById(ResourceTable.Id_dialog_sub_text);
        Button cancelButton = (Button) layoutComponent.findComponentById(ResourceTable.Id_cancelButton);
        Button confirmButton = (Button) layoutComponent.findComponentById(ResourceTable.Id_confirmButton);
        cancelButton.setClickedListener(component -> logoutDialog.destroy());
        confirmButton.setClickedListener((component -> terminateAbility()));

        dialogText.setText(ResourceTable.String_warning);
        dialogSubText.setText(ResourceTable.String_confirm_logout);
        logoutDialog
                .setContentCustomComponent(layoutComponent)
                .setTransparent(true)
                .setSize(AttrHelper.vp2px(300, this), DirectionalLayout.LayoutConfig.MATCH_CONTENT);

        logoutDialog.show();
    }

    /**
     * The showProgress, when loginButton clicked, the dialog show progress
     *
     * @param show show, show or hide the dialog
     */
    private void showProgress(final boolean show) {
        // Instance the dialog when dialog is null
        if (statusDialog == null) {
            statusDialog = new CommonDialog(this);

            // Get circleProgress animation
            Component circleProgress = drawCircleProgress(AttrHelper.vp2px(6, this), AttrHelper.vp2px(3, this));
            statusDialog
                    .setContentCustomComponent(circleProgress)
                    .setTransparent(true)
                    .setSize(
                            DirectionalLayout.LayoutConfig.MATCH_CONTENT, DirectionalLayout.LayoutConfig.MATCH_CONTENT);
        }

        // Show or hide the dialog
        if (show) {
            statusDialog.show();
        } else {
            statusDialog.destroy();
            statusDialog = null;
        }
    }

    /**
     * The drawCircleProgress, draw circle progress function
     *
     * @param maxRadius maxRadius,the radius of animation for the max circle
     * @param minRadius minRadius,the radius of animation for the min circle
     * @return The component of the animation
     */
    private Component drawCircleProgress(int maxRadius, int minRadius) {
        final int circleNum = 12;

        // Init the paint
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.FILL_STYLE);
        paint.setColor(Color.WHITE);

        // Init the component
        Component circleProgress = new Component(this);
        circleProgress.setComponentSize(AttrHelper.vp2px(100, this), AttrHelper.vp2px(100, this));
        circleProgress.setBackground(new ShapeElement(this, ResourceTable.Graphic_dialog_bg));

        // Draw the animation
        circleProgress.addDrawTask(
                (component, canvas) -> {
                    // Reset when a round
                    if (roateNum == circleNum) {
                        roateNum = 0;
                    }

                    // Rotate the canvas
                    canvas.rotate(30 * roateNum, (float) (component.getWidth() / 2), (float) (component.getHeight() / 2));
                    roateNum++;
                    int radius = (Math.min(component.getWidth(), component.getHeight())) / 2 - maxRadius;
                    float radiusIncrement = (float) (maxRadius - minRadius) / circleNum;
                    double angle = 2 * Math.PI / circleNum;

                    // Draw the small circle
                    for (int i = 0; i < circleNum; i++) {
                        float x = (float) (component.getWidth() / 2 + Math.cos(i * angle) * radius);
                        float y = (float) (component.getHeight() / 2 - Math.sin(i * angle) * radius);
                        paint.setAlpha((1 - (float) i / circleNum));
                        canvas.drawCircle(x, y, maxRadius - radiusIncrement * i, paint);
                    }

                    // Refresh the component delay
                    getUITaskDispatcher()
                            .delayDispatch(
                                    circleProgress::invalidate,
                                    150);
                });
        return circleProgress;
    }

    /**
     * The showDetectDialog, show the result of detect: No internet connection \ No return
     *
     * @param situation:  No internet connection or No return
     */
    private void showDetectDialog(boolean situation) {
        // Init dialog
        CommonDialog loginDialog = new CommonDialog(this);
        // Get component from xml
        Component layoutComponent =
                LayoutScatter.getInstance(this).parse(ResourceTable.Layout_status_dialog, null, false);
        Text dialogText = (Text) layoutComponent.findComponentById(ResourceTable.Id_dialog_text);
        Text dialogSubText = (Text) layoutComponent.findComponentById(ResourceTable.Id_dialog_sub_text);

        if (situation) {
            dialogText.setText(ResourceTable.String_fail);
            dialogSubText.setText(ResourceTable.String_detectNoInternet);
        } else {
            dialogText.setText(ResourceTable.String_fail);
            dialogSubText.setText(ResourceTable.String_detectFail);
        }

        loginDialog
                .setContentCustomComponent(layoutComponent)
                .setTransparent(true)
                .setSize(AttrHelper.vp2px(300, this), DirectionalLayout.LayoutConfig.MATCH_CONTENT)
                .setDuration(1500);

        loginDialog.show();
    }

    /**
     * Jump to DetectAbilityHistorySlice
     */
    private void startAbilitySlice(Component component) {
        Intent intent = new Intent();
        Operation operation = new Intent.OperationBuilder().withDeviceId("").withAction(ACTION)
                .withBundleName(getBundleName())
                .withAbilityName(DetectAbility.class.getName())
                .build();
        intent.setOperation(operation);
        startAbility(intent);
    }

    @Override
    protected void onStart(Intent intent) {
        super.onStart(intent);
        super.setUIContent(ResourceTable.Layout_detect_ability_main_slice);
        String[] permissions = {"ohos.permission.WRITE_USER_STORAGE", "ohos.permission.READ_USER_STORAGE"};
        requestPermissionsFromUser(permissions, REQUEST_CODE);
        initComponents();
        initListener();
    }

    @Override
    public void onActive() {
        super.onActive();
    }

    @Override
    public void onForeground(Intent intent) {
        super.onForeground(intent);
    }

    @Override
    protected void onAbilityResult(int requestCode, int resultCode, Intent resultData) {
        if(requestCode==REQUEST_CODE)
        {
            if(resultData == null){ return; }
            HiLog.info(label,"imageGetUriString:"+resultData.getUriString());
            // Get uri corresponds to the chosen image
            String chooseImgUri=resultData.getUriString();
            HiLog.info(label,"imageGetScheme:"+chooseImgUri.substring(chooseImgUri.lastIndexOf('/')));

            // Define DataAbilityHelper
            DataAbilityHelper helper=DataAbilityHelper.creator(getContext());
            // Get ID corresponds to the chosen image
            String chooseImgId=null;

            // If a file chosen - content://com.android.providers.media.documents/document/image%3A{ID}
            // If an image in album chosen - content://media/external/images/media/{ID}
            // Judge whether it is chosen from files or album
            if (chooseImgUri.lastIndexOf("%3A") != -1) {
                chooseImgId = chooseImgUri.substring(chooseImgUri.lastIndexOf("%3A") + 3);
            } else {
                chooseImgId = chooseImgUri.substring(chooseImgUri.lastIndexOf('/') + 1);
            }
            // Set uri corresponds to the chosen image into DataAbilityHelper
            Uri chosenImgUri = Uri.appendEncodedPathToUri(AVStorage.Images.Media.EXTERNAL_DATA_ABILITY_URI, chooseImgId);
            HiLog.info(label, "imagePath:" + chosenImgUri.toString());
            ImageSource imageSource = null;
            try {
                // Read file
                FileDescriptor fd = helper.openFile(chosenImgUri, "r");
                InputStream is = helper.obtainInputStream(chosenImgUri);
                imageSource = ImageSource.create(fd, null);
                imgStr = Base64Util.imageInputStreamToBase64(is);
                // Create pixelMap
                PixelMap pixelMap = imageSource.createPixelmap(null);
                // Set pixelMap to image component
                chosenImage.setPixelMap(pixelMap);
                // Validate detect button
                detectButton.setEnabled(true);
                detectButton.setBackground(new ShapeElement(this, ResourceTable.Graphic_btn_background_can));
                getUITaskDispatcher().asyncDispatch(() -> resultText.setText(""));
            } catch (Exception e) {
                e.printStackTrace();
            } finally {
                if (imageSource != null) {
                    imageSource.release();
                }
            }

        }
    }
}
