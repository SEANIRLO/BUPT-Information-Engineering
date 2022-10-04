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
import com.example.bburecog.ResourceTable;
import com.example.bburecog.utils.ElementUtil;
import com.example.bburecog.utils.Toast;
import com.example.bburecog.DetectAbility;
import ohos.aafwk.ability.AbilitySlice;
import ohos.aafwk.content.Intent;
import ohos.aafwk.content.Operation;
import ohos.agp.components.*;
import ohos.agp.components.element.ShapeElement;
import ohos.agp.render.Paint;
import ohos.agp.utils.Color;
import ohos.agp.window.dialog.CommonDialog;
import ohos.app.dispatcher.task.TaskPriority;
import ohos.eventhandler.EventHandler;
import ohos.eventhandler.EventRunner;
import ohos.eventhandler.InnerEvent;
import ohos.hiviewdfx.HiLogLabel;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * LoginAbilityMainSlice
 */
public class LoginAbilityMainSlice extends AbilitySlice {
    // HiLog Settings - Printing from emulator is not available
    private static final String TAG = LoginAbilityMainSlice.class.getSimpleName();
    private static final HiLogLabel LABEL_LOG = new HiLogLabel(3, 0xD000F00, TAG);

    // Preset userInfo - not connected to DB yet
    private static final String VALID_MAIL = "xh@163.com";
    private static final String VALID_PWD = "qwerty";

    // Event ID
    private static final int LOGIN_SUCCESS = 1000;
    private static final int LOGIN_FAIL = 1001;

    // Components
    private Text validMail;
    private Text validPassword;
    private TextField mailText;
    private TextField passwordText;
    private Button loginButton;
    private Image logo;
    private Text regisText;
    private Text aboutText;
    private CommonDialog commonDialog;

    // Current number of the animation for the circle progress
    private int roateNum = 0;

    // Instance a EventHandler get the result of login
    private final EventHandler loginEventHandler =
            new EventHandler(EventRunner.getMainEventRunner()) {
                @Override
                protected void processEvent(InnerEvent event) {
                    super.processEvent(event);
                    showProgress(false);
                    switch (event.eventId) {
                        case LOGIN_SUCCESS:
                            showLoginDialog(true);
                            mailText.setText("");
                            passwordText.setText("");
                            Intent intent = new Intent();
                            Operation operation = new Intent.OperationBuilder().withDeviceId("")
                                    .withBundleName(getBundleName())
                                    .withAbilityName(DetectAbility.class.getName())
                                    .build();
                            intent.setOperation(operation);
                            startAbility(intent);
                            break;
                        case LOGIN_FAIL:
                            showLoginDialog(false);
                            passwordText.setText("");
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
        validMail = (Text) findComponentById(ResourceTable.Id_validMail);
        validPassword = (Text) findComponentById(ResourceTable.Id_validPassword);
        mailText = (TextField) findComponentById(ResourceTable.Id_mailText);
        passwordText = (TextField) findComponentById(ResourceTable.Id_passwordText);
        loginButton = (Button) findComponentById(ResourceTable.Id_login_button);
        loginButton.setEnabled(false);
        logo = (Image) findComponentById(ResourceTable.Id_logo);
        regisText = (Text) findComponentById(ResourceTable.Id_regisText);
        aboutText = (Text) findComponentById(ResourceTable.Id_aboutText);
    }

    /**
     * The initListener, set listener of component
     */
    private void initListener() {
        mailText.addTextObserver(
                (text, var, i1, i2) -> {
                    validMail.setVisibility(Component.HIDE);
                    validPassword.setVisibility(Component.HIDE);
                });
        passwordText.addTextObserver(this::onTextUpdated);
        loginButton.setClickedListener(component -> login(mailText.getText(), passwordText.getText()));
        regisText.setClickedListener(
                component ->
                        Toast.makeToast(
                                        LoginAbilityMainSlice.this,
                                        getString(ResourceTable.String_clickedRegister),
                                        Toast.TOAST_SHORT)
                                .show());

        aboutText.setClickedListener(
                component -> present(new LoginAbilityAboutSlice(), new Intent()));
    }

    /**
     * Validate the login button and the hint text
     */
    private void onTextUpdated(String text, int var, int i1, int i2) {
        if (text != null && !text.isEmpty()) {
            loginButton.setEnabled(true);
            loginButton.setBackground(new ShapeElement(this, ResourceTable.Graphic_btn_background_can));
        } else {
            loginButton.setEnabled(false);
            loginButton.setBackground(new ShapeElement(this, ResourceTable.Graphic_btn_background));
        }
        validMail.setVisibility(Component.HIDE);
        validPassword.setVisibility(Component.HIDE);
    }

    /**
     * The mailValid, valid the mail format local
     *
     * @param mail mail,the text of mail
     * @return whether valid of the mail
     */
    private boolean mailValid(String mail) {
        return mail.matches("^[a-z0-9A-Z]+[- |a-z0-9A-Z._]+@([a-z0-9A-Z]+(-[a-z0-9A-Z]+)?\\.)+[a-z]{2,}$");
    }

    /**
     * The passwordValid, valid the password local
     *
     * @param password password,the text of password
     * @return whether valid of the password
     */
    private boolean passwordValid(String password) {
        return password.length() >= 6 && password.length() <= 16;
    }

    /**
     * The login, LOCALLY login action when clicked loginButton
     * First, valid mail and password local
     * Second, show the login dialog
     * Third, a thread for valid the login info simulate local
     * Fourth, sendEvent whether login success or fail
     *
     * @param mail     mail,the text of mail
     * @param password password,the text of password
     */
    private void login(final String mail, final String password) {
        validMail.setVisibility(Component.HIDE);
        validPassword.setVisibility(Component.HIDE);

        if (!mailValid(mail)) {
            validMail.setVisibility(Component.VISIBLE);
        } else if (!passwordValid(password)) {
            validPassword.setVisibility(Component.VISIBLE);
        } else {
            showProgress(true);
            // A thread for valid the login info
            getGlobalTaskDispatcher(TaskPriority.DEFAULT)
                    .asyncDispatch(
                            () -> {
                                try {
                                    Thread.sleep(1000);
                                } catch (InterruptedException e) {
                                    Logger.getLogger(ElementUtil.class.getName()).log(Level.SEVERE, e.getMessage());
                                }

                                // SendEvent whether login success or fail
                                if (mail.equals(VALID_MAIL) && password.equals(VALID_PWD)) {
                                    loginEventHandler.sendEvent(LOGIN_SUCCESS);
                                } else {
                                    loginEventHandler.sendEvent(LOGIN_FAIL);
                                }
                            });

        }
    }

    /**
     * The showProgress, when loginButton clicked, the dialog show progress
     *
     * @param show show, show or hide the dialog
     */
    private void showProgress(final boolean show) {
        // Instance the dialog when dialog is null
        if (commonDialog == null) {
            commonDialog = new CommonDialog(this);

            // Get circleProgress animation
            Component circleProgress = drawCircleProgress(AttrHelper.vp2px(6, this), AttrHelper.vp2px(3, this));
            commonDialog
                    .setContentCustomComponent(circleProgress)
                    .setTransparent(true)
                    .setSize(
                            DirectionalLayout.LayoutConfig.MATCH_CONTENT, DirectionalLayout.LayoutConfig.MATCH_CONTENT);
        }

        // Show or hide the dialog
        if (show) {
            commonDialog.show();
        } else {
            commonDialog.destroy();
            commonDialog = null;
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
     * The showLoginDialog, show the result of login whether success or fail
     *
     * @param success: success or fail
     */
    private void showLoginDialog(boolean success) {
        // Init dialog
        CommonDialog loginDialog = new CommonDialog(this);
        // Get component from xml
        Component layoutComponent =
                LayoutScatter.getInstance(this).parse(ResourceTable.Layout_status_dialog, null, false);
        Text dialogText = (Text) layoutComponent.findComponentById(ResourceTable.Id_dialog_text);
        Text dialogSubText = (Text) layoutComponent.findComponentById(ResourceTable.Id_dialog_sub_text);

        if (success) {
            dialogText.setText(ResourceTable.String_success);
            dialogSubText.setText(ResourceTable.String_loginSuccess);
        } else {
            dialogText.setText(ResourceTable.String_fail);
            dialogSubText.setText(ResourceTable.String_loginFail);
        }

        loginDialog
                .setContentCustomComponent(layoutComponent)
                .setTransparent(true)
                .setSize(AttrHelper.vp2px(300, this), DirectionalLayout.LayoutConfig.MATCH_CONTENT)
                .setDuration(1500);

        loginDialog.show();
    }


    @Override
    protected void onStart(Intent intent) {
        super.onStart(intent);
        super.setUIContent(ResourceTable.Layout_login_ability_main_slice);
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
}
